import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import sys

try:
    from simulator import PNPSimulatorAdaptiveLight
except ImportError:
    raise ImportError("오류")

def generate_mixed_scenario(total_duration_ms=400, dt=0.1):
    total_steps = int(total_duration_ms / dt)
    I_full = np.zeros(total_steps)
    current_idx = 0
    
    SAFE_MAX_CURRENT = 20.0 
    
    while current_idx < total_steps:
        remaining_steps = total_steps - current_idx
        
        if remaining_steps < 500: block_steps = remaining_steps
        else: block_steps = int(np.random.uniform(50, 150) / dt)
            
        if current_idx + block_steps > total_steps:
            block_steps = total_steps - current_idx
            
        scenario = np.random.choice(['silence', 'step', 'ramp', 'osc', 'noise_burst'], 
                                    p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        t_local = np.linspace(0, block_steps * dt, block_steps)
        I_block = np.zeros(block_steps)
        
        if scenario == 'silence':
            I_block = np.zeros(block_steps)
            
        elif scenario == 'step':
            amp = np.random.uniform(2.0, 10.0)
            I_block[:] = amp
            
        elif scenario == 'ramp':
            start_amp = np.random.uniform(0, 4.0)
            end_amp = np.random.uniform(4.0, 8.0)
            if np.random.rand() > 0.5:
                I_block = np.linspace(start_amp, end_amp, block_steps)
            else:
                I_block = np.linspace(end_amp, start_amp, block_steps)
                
        elif scenario == 'osc':
            freq = np.random.uniform(4, 30)
            bias = np.random.uniform(1.0, 7.0)
            max_amp = SAFE_MAX_CURRENT - bias
            amp = np.random.uniform(0.5, max_amp) 
            phase = np.random.uniform(0, 2*np.pi)
            I_block = bias + amp * np.sin(2 * np.pi * freq * t_local / 1000.0 + phase)
            
        elif scenario == 'noise_burst':
            noise = np.random.normal(0, 1, block_steps)
            filtered_noise = gaussian_filter1d(noise, sigma=20) 
            filtered_noise = filtered_noise / (np.std(filtered_noise) + 1e-9) 
            I_block = filtered_noise * np.random.uniform(1.0, 2.5) + np.random.uniform(1.0, 2.0)

        I_full[current_idx : current_idx + block_steps] = I_block
        current_idx += block_steps

    # bg_noise가 너무 크거나 guassian filter의 sigma가 작으면 simulation이 불안정해짐
    bg_noise = np.random.normal(0, 0.1, size=total_steps)
    I_final = gaussian_filter1d(I_full, sigma=70) + bg_noise
    
    I_final = np.clip(I_final, -1.0, SAFE_MAX_CURRENT)
    
    return I_final

def generate_and_save_raw_data_adaptive(n_samples=2000, duration_ms=400, report_dt=0.1, filename='400ms_data.npz'):

    target_length = int(duration_ms / report_dt)
    
    valid_raw_X = []
    valid_raw_Y = []
    
    pbar = tqdm(range(n_samples), ncols=100)
    
    for i in pbar:
        sim = PNPSimulatorAdaptiveLight(nx=32, ny=32)
        
        I_input_array = generate_mixed_scenario(total_duration_ms=duration_ms, dt=report_dt)
        
        v_rec = []
        k_rec = []
        
        sim_time = 0.0
        
        for t_idx in range(target_length):
            target_next_time = (t_idx + 1) * report_dt
            current_val = I_input_array[t_idx]
            
            while sim_time < target_next_time:
                remaining_time = target_next_time - sim_time
                

                vm_curr = sim.Vm
                
                if vm_curr > -55.0:  
                    max_dt = 0.005  # 5 us
                elif vm_curr < -80.0:
                    max_dt = 0.01   # 10 us
                else:
                    max_dt = 0.05   # 50 us
    
                dt_step = min(max_dt, remaining_time)

                if dt_step < 1e-6:
                    dt_step = remaining_time
    
                try:
                    sim.step(dt_step, ext_current=current_val)
                    sim_time += dt_step
                    
                    if sim.Vm > 100.0: 
                        sim.Vm = 100.0
                    elif sim.Vm < -100.0: 
                        sim.Vm = -100.0
                        
                except FloatingPointError:
                    v_rec = [] 
                    break
            
            if len(v_rec) == 0 and t_idx > 0: break

            v_rec.append(sim.Vm)
            k_val = np.mean(sim.c['K'][:, sim.mem_start_y-1])
            k_rec.append(k_val)
            
        if len(v_rec) == target_length:
            v_arr = np.array(v_rec)
            k_arr = np.array(k_rec)
            
            v_n = v_arr + np.random.normal(0, 1.0, size=target_length)
            k_n = k_arr 
            
            dv = np.gradient(v_n) / report_dt 
            dk = np.gradient(k_n) / report_dt
            
            x_sample = np.stack([v_n, k_n, dv, dk], axis=-1).astype(np.float32)
            y_sample = I_input_array.reshape(-1, 1).astype(np.float32)
            
            valid_raw_X.append(x_sample)
            valid_raw_Y.append(y_sample)
            
            pbar.set_postfix(saved=len(valid_raw_X))
            
    X_final = np.array(valid_raw_X, dtype=np.float32)
    Y_final = np.array(valid_raw_Y, dtype=np.float32)
    
    # 저장 경로 설정 (Kaggle 환경이 아니면 현재 폴더로 수정 가능)
    save_path = filename
    if os.path.exists('/kaggle/working/'):
        save_path = f'/kaggle/working/{filename}'
    
    np.savez_compressed(save_path, raw_X=X_final, raw_Y=Y_final)
    print(f"저장 위치: {save_path}")
    print(f"Shape - X: {X_final.shape}, Y: {Y_final.shape}")
    
    return X_final, Y_final

#if __name__ == "__main__":
#    raw_X, raw_Y = generate_and_save_raw_data_adaptive(n_samples=2000, duration_ms=400)

## 샘플 시각화 코드

def visualize_mixed_samples(X, Y, dt=0.1, n_samples=3):
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        t = np.arange(X.shape[1]) * dt
        v = X[idx, :, 0]   
        i_stim = Y[idx, :, 0] 
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        ax[0].plot(t, i_stim, 'r-', label='Input Current (Mixed)')
        ax[0].set_ylabel('Current (uA/cm2)')
        ax[0].set_title(f"Sample {idx}: Stimulus Scenario")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc='upper right')
        
        ax[1].plot(t, v, 'b-', label='Membrane Voltage')
        ax[1].set_ylabel('Voltage (mV)')
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_title(f"Sample {idx}: Neural Response")
        ax[1].grid(True, alpha=0.3)
        
        if np.max(v) > 0:
            ax[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.show()

# 시각화 실행
#visualize_mixed_samples(raw_X, raw_Y)