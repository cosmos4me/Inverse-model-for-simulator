# Inverse-model-for-simulator

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Flow Matching](https://img.shields.io/badge/Generative-Flow%20Matching-blue)
![Adaptive Solver](https://img.shields.io/badge/Solver-Adaptive%20RK45-green)

This repository contains an implementation of **Inverse Flow Matching** for reconstructing neural stimulation currents from voltage responses. By solving the inverse problem of the Hodgkin-Huxley (or similar) neuronal dynamics, this model predicts the precise input current ($I$) required to generate a target voltage trace ($V$).

The model utilizes a **1D U-Net** architecture conditioned via **FiLM (Feature-wise Linear Modulation)** and is trained with a **Hybrid Spectral Loss** for high-fidelity signal reconstruction.

## 🌟 Key Features

- **Inverse Flow Matching:** Generates continuous-time trajectories from noise to target stimulus currents using ODE-based generative modeling.
- **FiLM Conditioning:** The model is conditioned on Voltage ($V$) and membrane parameters ($K, \dot{V}, \dot{K}$) using Feature-wise Linear Modulation layers for precise control.
- **Adaptive ODE Solver:** Inference is performed using `scipy.integrate.solve_ivp` (RK45), allowing for error-controlled, adaptive step sizes for maximum accuracy.
- **Advanced Spectral Loss:** Training utilizes a composite loss function:
  - **MSE Loss:** For signal fidelity.
  - **Gradient Loss:** To capture sharp spikes and rapid changes.
  - **FFT Loss:** To ensure accuracy in the frequency domain.
  - **Sparsity (L1) Loss:** To suppress noise.
- **EMA (Exponential Moving Average):** Stabilizes training and improves validation performance.
- **Efficient Attention:** Incorporates PyTorch 2.0+ FlashAttention (Scaled Dot Product Attention) for computational efficiency.

## 📂 Project Structure

```bash
.
├── model.py          # U-Net Architecture, FiLM, Attention blocks, and Loss functions
├── train.py          # Main training loop, EMA, and Evaluation pipeline
├── preprocess.py     # Data loading, Normalization, and PyTorch Dataset creation
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
