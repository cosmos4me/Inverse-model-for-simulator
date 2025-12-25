import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
import os

from preprocess import prepare_dataloaders
from model import UNetInverseFlowMatching, AdvancedSpectralLoss

# EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Flow Sampler 
class FlowSampler:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def sample(self, condition, steps=50):
        self.model.eval()
        batch_size = condition.shape[0]
        length = condition.shape[-1]
        
        x_current = torch.randn(batch_size, 1, length).to(self.device)
        
        dt = 1.0 / steps
        time_steps = torch.linspace(0, 1, steps).to(self.device)
        
        for t in time_steps:
            t_batch = torch.ones(batch_size).to(self.device) * t
            velocity = self.model(t_batch, x_current, condition)
            x_current = x_current + velocity * dt
            
        return x_current

    @torch.no_grad()
    def sample_adaptive(self, condition, rtol=1e-5, atol=1e-5):
        self.model.eval()
        batch_size, _, length = condition.shape
        
        y0 = torch.randn(batch_size, 1, length).to(self.device).cpu().numpy().flatten()
        
        condition = condition.to(self.device)

        def ode_func(t, y_flat):
            y_tensor = torch.from_numpy(y_flat).reshape(batch_size, 1, length).float().to(self.device)
            t_tensor = torch.ones(batch_size).to(self.device) * t
            
            with torch.no_grad():
                velocity = self.model(t_tensor, y_tensor, condition)
            
            return velocity.cpu().numpy().flatten()

        sol = solve_ivp(
            ode_func, 
            t_span=(0, 1), 
            y0=y0, 
            method='RK45', 
            rtol=rtol, 
            atol=atol
        )
        result_flat = sol.y[:, -1]
        result_tensor = torch.from_numpy(result_flat).reshape(batch_size, 1, length).to(self.device)
        
        return result_tensor

# Utils & Visualization
def inverse_transform(tensor, mean, std):
    device = tensor.device
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device).float()
        std = torch.tensor(std, device=device).float()
    return tensor * std + mean

def plot_loss_curve(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title("Learning Curve (Hybrid Loss + EMA)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_predictions(model, test_loader, stats, device, n_samples=3):
    sampler = FlowSampler(model, device)
    condition, target_batch = next(iter(test_loader))
    condition = condition[:n_samples].to(device)
    target_batch = target_batch[:n_samples].to(device)

    pred_norm = sampler.sample_adaptive(condition, rtol=1e-3, atol=1e-3)

    y_mean, y_std = stats['y_mean'], stats['y_std']
    pred_real = inverse_transform(pred_norm, y_mean, y_std).cpu().numpy()
    target_real = inverse_transform(target_batch, y_mean, y_std).cpu().numpy()
    
    v_trace = inverse_transform(condition[:, 0:1, :], stats['x_mean'][0], stats['x_std'][0]).cpu().numpy()
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
    time_axis = np.arange(pred_real.shape[2]) * 0.1
    
    for k in range(n_samples):
        ax1 = axes[k, 0]
        ax1.plot(time_axis, v_trace[k, 0], 'k-', alpha=0.7, label='Input Voltage')
        ax1.set_title(f"Sample {k+1}: Neural Input")
        ax1.set_ylabel("Voltage (mV)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[k, 1]
        ax2.plot(time_axis, target_real[k, 0], 'b--', linewidth=2, label='True Stimulus')
        ax2.plot(time_axis, pred_real[k, 0], 'r-', alpha=0.8, label='Predicted Stimulus')
        ax2.set_title(f"Sample {k+1}: Stimulus Reconstruction")
        ax2.set_ylabel("Current (uA/cm2)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        mse = np.mean((target_real[k, 0] - pred_real[k, 0])**2)
        ax2.text(0.05, 0.95, f"MSE: {mse:.4f}", transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

# Training Engine
class FlowMatchingEngine:
    def __init__(self, model, device, stats):
        self.model = model.to(device)
        self.device = device
        self.stats = stats
        
        self.criterion = AdvancedSpectralLoss(
            alpha_fft=0.1, 
            alpha_l1=0.01, 
            alpha_grad=2.0  
        ).to(device)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.ema = EMA(self.model, decay=0.999)

    def compute_loss(self, x_1, condition):
        batch_size = x_1.shape[0]
        t = torch.rand(batch_size, device=self.device)
        x_0 = torch.randn_like(x_1)
        
        t_view = t.view(-1, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1
        target_v = x_1 - x_0
        
        pred_v = self.model(t, x_t, condition)
        
        return self.criterion(pred_v, target_v)

    def train(self, train_loader, val_loader=None, epochs=50):
        print(f">>> Advanced Training Started (Device: {self.device}) | EMA: ON")
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 1. Train
            self.model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for condition, target in pbar:
                condition, target = condition.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.compute_loss(target, condition)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                self.ema.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            self.scheduler.step()
            
            avg_val_loss = 0
            if val_loader:
                self.ema.apply_shadow()
                
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for condition, target in val_loader:
                        condition, target = condition.to(self.device), target.to(self.device)
                        loss = self.compute_loss(target, condition)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                self.ema.restore()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_msg = f" | Val Loss (EMA): {avg_val_loss:.6f}" if val_loader else ""
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_msg}")
                
        return history

# Main Pipeline
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current Device: {device}")

    RAW_DATA_PATH = '/kaggle/input/real-data/400ms_data.npz'
    
    train_loader, val_loader, test_loader, stats = prepare_dataloaders(
        raw_path=RAW_DATA_PATH,
        batch_size=32,
        dt=0.1
    )

    model = UNetInverseFlowMatching(input_dim=1, cond_dim=4, hidden_dim=128).to(device)
    engine = FlowMatchingEngine(model, device, stats)

    history = engine.train(train_loader, val_loader=val_loader, epochs=100)

    engine.ema.apply_shadow()
    print(">>> EMA weights applied for final evaluation.")

    plot_loss_curve(history)                  
    evaluate_predictions(model, test_loader, stats, device, n_samples=3)

    save_path = "inverse_model.pth"
    torch.save(model.state_dict(), save_path)
