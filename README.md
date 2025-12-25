# Inverse-model-for-simulator

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Flow Matching](https://img.shields.io/badge/Generative-Flow%20Matching-blue)
![Adaptive Solver](https://img.shields.io/badge/Solver-Adaptive%20RK45-green)

This repository contains an implementation of **Inverse Flow Matching** for reconstructing neural stimulation currents from voltage responses. By solving the inverse problem of the Hodgkin-Huxley and Poisson-Nernst-Plank neuronal dynamics, this model predicts the precise input current ($I$) required to generate a target voltage trace ($V$).

The model utilizes a **1D U-Net** architecture conditioned via **FiLM (Feature-wise Linear Modulation)** and is trained with a **Hybrid Spectral Loss** for high-fidelity signal reconstruction.


## 📊 Results (using test set / split entire data set to train/val/test)

| Metric | Value | 
| :--- | :--- |
| **Global $R^2$ Score** | **0.9937** |
| **Avg Correlation** | **0.9952** |
| **RMSE** | 0.2494 $\mu A/cm^2$ | 
| **MAE** | 0.1783 $\mu A/cm^2$ |

## 🛡️ Robustness Analysis
I evaluated the model's stability under low Signal-to-Noise Ratio (SNR) conditions to simulate real-world recording artifacts.

**Experiment Setup:**
- **Noise Injection:** Gaussian noise ($\sigma=0.4$) was added to the normalized voltage traces. This represents a significant degradation of the input signal.
- **Preprocessing:** The derivative ($dV/dt$) was re-calculated from the noisy input to test the model's sensitivity to high-frequency fluctuations.

**Observations:**
![Robustness Test Result](./images/robust1.png)


## 📂 Project Structure

```bash
.
├── model.py          
├── train.py         
├── preprocess.py
├── simulator.py
├── raw_data_generation.py
├── requirements.txt
└── README.md         
