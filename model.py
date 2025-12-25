import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from torchdiffeq import odeint

class AdvancedSpectralLoss(nn.Module):
    def __init__(self, alpha_fft=0.1, alpha_l1=0.01, alpha_grad=1.0):
        super().__init__()
        self.alpha_fft = alpha_fft
        self.alpha_l1 = alpha_l1
        self.alpha_grad = alpha_grad

    def forward(self, pred, target):
        loss_mse = F.mse_loss(pred, target)

        pred_grad = pred[..., 1:] - pred[..., :-1]
        target_grad = target[..., 1:] - target[..., :-1]
        loss_grad = F.mse_loss(pred_grad, target_grad)
        
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        loss_fft = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        loss_l1 = torch.mean(torch.abs(pred))
        
        total = loss_mse + \
                (self.alpha_grad * loss_grad) + \
                (self.alpha_fft * loss_fft) + \
                (self.alpha_l1 * loss_l1)
                
        return total

# Building Blocks (FiLM, Attention, Positional Embedding)
class FiLM(nn.Module):

    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, condition):
        cond_perm = condition.permute(0, 2, 1) 
        
        style = self.adaptor(cond_perm) # (B, L, 2*C)
        style = style.permute(0, 2, 1)  # (B, 2*C, L)
        
        gamma, beta = torch.chunk(style, 2, dim=1)
        
        return (1.0 + gamma) * x + beta

class FiLMResBlock1D(nn.Module):
    def __init__(self, channels, cond_dim, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.film1 = FiLM(cond_dim, channels)
        
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)
        self.film2 = FiLM(cond_dim, channels)
        
        self.act = nn.Mish()
        
        self.residual_proj = nn.Identity()

    def forward(self, x, condition):
        residual = self.residual_proj(x)
        
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.film1(h, condition)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.film2(h, condition)
        h = self.act(h)
        
        return h + residual

class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = C // self.num_heads
        
        q = q.view(B, self.num_heads, head_dim, L).transpose(-1, -2) # (B, H, L, D)
        k = k.view(B, self.num_heads, head_dim, L).transpose(-1, -2)
        v = v.view(B, self.num_heads, head_dim, L).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).reshape(B, L, C).transpose(1, 2)
        
        out = self.proj(out)
        return x + out

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        self.register_buffer('pe', pe.transpose(0, 1).unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, :x.size(-1)]

# Main Model
class UNetInverseFlowMatching(nn.Module):
    def __init__(self, input_dim=1, cond_dim=4, hidden_dim=64):
        super().__init__()
        
        ch1 = hidden_dim
        ch2 = hidden_dim * 2
        ch3 = hidden_dim * 4
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, ch1), nn.Mish(), nn.Linear(ch1, ch1)
        )
        
        # Input Projection
        self.input_proj = nn.Conv1d(input_dim, ch1, 3, padding=1)

        # Positional Encoding
        self.pos_emb = SinusoidalPositionalEmbedding(ch1)
        
        # --- Encoder ---
        # Level 1 (L)
        self.down1 = FiLMResBlock1D(ch1, cond_dim, dilation=1)
        self.pool1 = nn.Conv1d(ch1, ch1, 3, stride=2, padding=1)
        
        # Level 2 (L/2)
        self.enc_conv2 = nn.Conv1d(ch1, ch2, 1)
        self.down2 = FiLMResBlock1D(ch2, cond_dim, dilation=2)
        self.pool2 = nn.Conv1d(ch2, ch2, 3, stride=2, padding=1)
        
        # --- Bottleneck (L/4) --- 
        self.enc_conv3 = nn.Conv1d(ch2, ch3, 1)
        self.bot1 = FiLMResBlock1D(ch3, cond_dim, dilation=2) 
        self.bot2 = FiLMResBlock1D(ch3, cond_dim, dilation=2)
        self.attn_bot = AttentionBlock1D(ch3)
        
        # --- Decoder ---
        # Level 2 Upsample (L/4 -> L/2)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec_conv2 = nn.Conv1d(ch3 + ch2, ch2, 1)
        self.dec_block2 = FiLMResBlock1D(ch2, cond_dim, dilation=2) 
        
        # Level 1 Upsample (L/2 -> L)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec_conv1 = nn.Conv1d(ch2 + ch1, ch1, 1)
        self.dec_block1 = FiLMResBlock1D(ch1, cond_dim, dilation=1)
        
        # --- Output ---
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch1),
            nn.Mish(),
            nn.Conv1d(ch1, input_dim, 3, padding=1)
        )

    def forward(self, t, x, condition):
        # Time Embedding
        if t.dim() == 0: t = t.unsqueeze(0).repeat(x.shape[0])
        t_emb = self.time_mlp(t.view(-1, 1)).unsqueeze(-1)
        
        # Input Setup
        h = self.input_proj(x)
        
        h = self.pos_emb(h) + t_emb
        
        # --- Encoder ---
        # Level 1
        h1 = self.down1(h, condition)
        
        # Level 2
        h_down = self.pool1(h1)
        cond_down = F.interpolate(condition, scale_factor=0.5, mode='linear')
        
        h2 = self.enc_conv2(h_down)
        h2 = self.down2(h2, cond_down)
        
        # --- Bottleneck ---
        h_bot = self.pool2(h2)
        cond_bot = F.interpolate(condition, scale_factor=0.25, mode='linear')
        
        h3 = self.enc_conv3(h_bot)
        h3 = self.bot1(h3, cond_bot)
        h3 = self.attn_bot(h3)
        h3 = self.bot2(h3, cond_bot)
        
        # --- Decoder ---
        # Level 2
        h_up2 = self.up2(h3)
        if h_up2.shape[-1] != h2.shape[-1]:
            h_up2 = F.interpolate(h_up2, size=h2.shape[-1], mode='linear')
            
        h_cat2 = torch.cat([h_up2, h2], dim=1)
        h_dec2 = self.dec_conv2(h_cat2)
        h_dec2 = self.dec_block2(h_dec2, cond_down)
        
        # Level 1
        h_up1 = self.up1(h_dec2)
        if h_up1.shape[-1] != h1.shape[-1]:
            h_up1 = F.interpolate(h_up1, size=h1.shape[-1], mode='linear')
            
        h_cat1 = torch.cat([h_up1, h1], dim=1)
        h_dec1 = self.dec_conv1(h_cat1)
        h_dec1 = self.dec_block1(h_dec1, condition)
        
        return self.output_proj(h_dec1)
