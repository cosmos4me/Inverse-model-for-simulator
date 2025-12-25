import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

class NeuroDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.from_numpy(x_data).float().permute(0, 2, 1)
        self.y = torch.from_numpy(y_data).float().permute(0, 2, 1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def prepare_dataloaders(raw_path='/kaggle/input/real-data/400ms_data.npz', batch_size=32, dt=0.1):
    print(">>> 데이터 전처리: [Input: Raw V, K] + [Hint: Smooth dV, dK]")
    
    # 파일 경로 예외 처리
    if not os.path.exists(raw_path):
        possible_files = ['400ms_data.npz', 'neuro_raw_data.npz']
        found = False
        for f in possible_files:
            if os.path.exists(f):
                raw_path = f
                found = True
                print(f"   Notice: {f} 파일을 찾아서 사용합니다.")
                break
        if not found:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {raw_path}")
            
    try:
        data = np.load(raw_path)
        X_origin = data['raw_X'] 
        Y = data['raw_Y']
    except Exception as e:
        raise RuntimeError(f"데이터 로딩 중 오류 발생: {e}")
    
    V_raw = X_origin[:, :, 0]
    K_raw = X_origin[:, :, 1]
    
    # Smoothing (Savitzky-Golay)
    V_smooth = savgol_filter(V_raw, window_length=15, polyorder=3, axis=1)
    K_smooth = savgol_filter(K_raw, window_length=15, polyorder=3, axis=1)
    
    # Recalculate Derivatives
    dV_clean = np.gradient(V_smooth, axis=1) / dt
    dK_clean = np.gradient(K_smooth, axis=1) / dt
    
    # Stack
    # Channel 0: V_raw    (노이즈 있음 - 모델이 실제 환경처럼 견뎌야 함)
    # Channel 1: K_raw    (노이즈 있음)
    # Channel 2: dV_clean (깔끔함 - 모델에게 주는 강력한 힌트/가이드)
    # Channel 3: dK_clean (깔끔함)
    X_final = np.stack([V_raw, K_raw, dV_clean, dK_clean], axis=-1).astype(np.float32)
    
    print("   Data Reconstruction Done.")

    X_train, X_temp, Y_train, Y_temp = train_test_split(X_final, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    
    x_mean = X_train.mean(axis=(0, 1))
    x_std = X_train.std(axis=(0, 1)) + 1e-6
    y_mean = Y_train.mean(axis=(0, 1))
    y_std = Y_train.std(axis=(0, 1)) + 1e-6
    
    def normalize(d, m, s): return (d - m) / s
    
    X_train_n = normalize(X_train, x_mean, x_std)
    X_val_n   = normalize(X_val, x_mean, x_std)
    X_test_n  = normalize(X_test, x_mean, x_std)
    Y_train_n = normalize(Y_train, y_mean, y_std)
    Y_val_n   = normalize(Y_val, y_mean, y_std)
    Y_test_n  = normalize(Y_test, y_mean, y_std)
    
    save_dir = '/kaggle/working/' if os.path.exists('/kaggle/working/') else '.'
    save_path = os.path.join(save_dir, 'neuro_processed_hybrid.npz')
    
    np.savez_compressed(save_path, 
                        x_train=X_train_n, y_train=Y_train_n,
                        x_val=X_val_n, y_val=Y_val_n,
                        x_test=X_test_n, y_test=Y_test_n,
                        stats={'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std})
    
    print(f"   Processed data saved to: {save_path}")

    train_loader = DataLoader(NeuroDataset(X_train_n, Y_train_n), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(NeuroDataset(X_val_n, Y_val_n), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(NeuroDataset(X_test_n, Y_test_n), batch_size=batch_size, shuffle=False)
    
    stats = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
    
    return train_loader, val_loader, test_loader, stats

if __name__ == "__main__":
    try:
        tr, va, te, st = prepare_dataloaders(batch_size=32)
        print(">>> Success! Loader sizes:", len(tr), len(va), len(te))
        print(">>> Stats:", st)
    except Exception as e:
        print(">>> Error during test run:", e)