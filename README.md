# Machine Learning-Based Passive Beamforming Optimization for RIS-Assisted Wireless Communication

## Overview

This project investigates the use of machine learning for passive beamforming optimization in a Reconfigurable Intelligent Surface (RIS)-assisted wireless communication system.

The system considers a Single-Input Single-Output (SISO) communication link in which the signal from the Base Station (BS) reaches the user through an RIS consisting of multiple reflecting elements.

The main objective is to learn a mapping from the cascaded BS-RIS-user channel to discrete RIS phase configurations and evaluate how closely the learned configuration approaches analytical and optimization-based reference methods.

The project combines concepts from:

- Wireless Communications
- Reconfigurable Intelligent Surfaces (RIS)
- Rayleigh Fading Channels
- Passive Beamforming
- Discrete Phase Optimization
- Machine Learning
- Communication System Performance Evaluation

---

## Problem Statement

An RIS can improve wireless communication performance by adjusting the phase shift introduced by each reflecting element.

For an RIS with N elements, the effective cascaded channel can be expressed as:

```text
h_eff = Σ(n=1 to N) [h_RU,n × φ_n × h_BR,n]
```

Where:

- `h_BR,n` is the Base Station-to-RIS channel for element n.
- `h_RU,n` is the RIS-to-user channel for element n.
- `φ_n` is the phase coefficient of RIS element n.
- `N` is the total number of RIS reflecting elements.

The RIS phase coefficients are constrained to a discrete set in the considered system.

For a 2-bit RIS, the available phase shifts are:

```text
F = {0, π/2, π, 3π/2}
```

The challenge is to select an appropriate phase for every RIS element to obtain a strong effective channel.

---

## System Model

The project considers a SISO BS-RIS-user system with the following parameters:

- Number of RIS elements: N = 16
- Channel model: Rayleigh fading
- Fading assumption: Quasi-static block fading
- Transmission: Narrowband
- Transmit power: P = 1
- Noise power: σ² = 1
- Direct BS-user link: Not considered in the initial system model

The BS-RIS and RIS-user channels are modeled as independent complex Gaussian random variables:

```text
h_BR,n ~ CN(0, 1)

h_RU,n ~ CN(0, 1)
```

The cascaded channel for each RIS element is:

```text
c_n = h_BR,n × h_RU,n
```

Therefore, the effective channel becomes:

```text
h_eff = Σ(n=1 to N) [c_n × φ_n]
```

The received Signal-to-Noise Ratio (SNR) is calculated as:

```text
SNR = (P × |h_eff|²) / σ²
```

The achievable communication rate is evaluated using the Shannon capacity expression:

```text
R = log₂(1 + SNR)
```

Where:

- `P` represents the transmit power.
- `|h_eff|²` represents the effective channel power gain.
- `σ²` represents the noise power.
- `R` represents the achievable rate in bits/s/Hz.

---

## RIS Phase Optimization

### 1. Continuous-Phase Reference

For continuous RIS phase control, the phase of each element can be selected to compensate for the phase of its cascaded channel.

The optimal phase angle is:

```text
θ_n* = -∠(c_n)
```

Therefore, the optimal phase coefficient is:

```text
φ_n* = exp(j × θ_n*)
```

This aligns the contributions from different RIS elements and provides coherent combining.

The continuous-phase solution serves as an ideal reference for evaluating the performance of discrete phase configurations.

### 2. Quantized Analytical Phase Selection

Real RIS implementations generally use a finite number of phase states.

For the 2-bit RIS used in this project, the available phase shifts are:

```text
F = {0, π/2, π, 3π/2}
```

The continuous optimal phase is quantized to the nearest available phase level.

The quantization operation can be expressed as:

```text
θ_n,quantized = Q(θ_n*)
```

Where `Q(.)` represents the nearest-phase quantization operation over the available discrete phase set.

This provides a simple analytical reference for evaluating the ML model.

### 3. Coordinate Descent Optimization

A Coordinate Descent (CD) method is also implemented as an optimization-based benchmark.

The algorithm optimizes one RIS element at a time.

For each RIS element:

1. All available discrete phase values are evaluated while keeping the other RIS phases fixed.
2. The phase producing the highest effective channel gain is selected.
3. The process is repeated for the remaining RIS elements.
4. Multiple iterations are performed until convergence or the maximum iteration limit is reached.

The objective is to maximize the effective channel power:

```text
Maximize: |h_eff|²

Subject to:

φ_n ∈ {1, j, -1, -j}
```

This provides a practical heuristic optimization reference without performing exhaustive search.

For N = 16 and four possible phase states per element, exhaustive search would require evaluating:

```text
Number of configurations = 4^16

                        = 4,294,967,296
```

This illustrates why heuristic optimization methods become useful as the number of RIS elements increases.

---

## Machine Learning Approach

The machine learning model learns the relationship between the cascaded channel and the corresponding discrete RIS phase configuration.

Instead of using the four separate channel vectors as the input, the final ML formulation directly uses the cascaded channel:

```text
c_n = h_BR,n × h_RU,n
```

### Input Feature Representation

The input features consist of the real and imaginary components of the cascaded channel:

```text
x = [
    Re(c_1), Re(c_2), ..., Re(c_N),
    Im(c_1), Im(c_2), ..., Im(c_N)
]
```

For N = 16, the total number of input features is:

```text
Number of input features = 2 × N

                         = 2 × 16

                         = 32
```

### Target Representation

The target contains one of four phase classes for each RIS element:

```text
y_n ∈ {0, 1, 2, 3}
```

The classes correspond to the following RIS phase configurations:

| Class | RIS Phase |
|:---:|:---:|
| 0 | 0 |
| 1 | π/2 |
| 2 | π |
| 3 | 3π/2 |

The training targets are generated from the analytical phase-selection rule followed by 2-bit phase quantization.

Therefore, the ML model is designed to **learn an approximation of the analytical phase-selection mapping**, rather than replacing the analytical solution with an unexplained black-box target.

---

## Neural Network Architecture

A shared neural network extracts features from the cascaded channel.

The architecture consists of the following layers:

```text
Input Layer: 32 Features
       |
       v
Fully Connected Layer: 256 Neurons
       |
       v
      ReLU
       |
       v
Fully Connected Layer: 128 Neurons
       |
       v
      ReLU
       |
       v
Fully Connected Layer: 64 Neurons
       |
       v
      ReLU
       |
       v
   16 Phase Heads
       |
       v
  4 Classes per Element
```

Each phase head predicts one of the four discrete phase classes for its corresponding RIS element.

Therefore, the network produces 16 individual phase predictions, one for each RIS reflecting element.

---

## Performance Evaluation

The learned RIS phase configuration is evaluated against the following reference methods:

1. **Continuous-Phase Optimization:** Ideal phase alignment without quantization constraints.
2. **Quantized Analytical Selection:** Analytical phase alignment followed by 2-bit phase quantization.
3. **Coordinate Descent:** Iterative optimization of discrete RIS phase configurations.
4. **Machine Learning Model:** Neural network-based prediction of discrete RIS phase configurations.

The performance can be evaluated using:

- Effective channel power gain: `|h_eff|²`
- Received SNR
- Achievable rate (bits/s/Hz)
- Performance gap relative to analytical and optimization-based references

These comparisons help assess how effectively the neural network learns the passive beamforming phase-selection strategy.

---

## Project Objectives

- Develop a mathematical model for an RIS-assisted SISO communication system.
- Implement Rayleigh fading channels for the BS-RIS and RIS-user links.
- Implement continuous and discrete RIS phase optimization.
- Develop a Coordinate Descent benchmark for discrete phase optimization.
- Generate training data using analytical phase-selection techniques.
- Train a neural network to predict discrete RIS phase configurations.
- Compare the ML-based approach with analytical and optimization-based reference methods.
- Evaluate communication performance using effective channel gain, SNR, and achievable rate.

---

## Technologies and Tools

- Python
- NumPy
- Matplotlib
- PyTorch
- Machine Learning
- Wireless Communication System Modeling

---

## Conclusion

This project explores the application of machine learning to passive beamforming optimization in RIS-assisted wireless communication systems.

By learning the relationship between cascaded channel coefficients and discrete RIS phase configurations, the proposed neural network aims to approximate analytical phase-selection decisions.

The performance of the learned model is evaluated against continuous-phase optimization, quantized analytical phase selection, and Coordinate Descent benchmarks.

The project provides an opportunity to study the intersection of wireless communications, optimization techniques, and machine learning for intelligent wireless environments.
