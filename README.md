\# Machine Learning-Based Passive Beamforming Optimization for RIS-Assisted Wireless Communication



\## Overview



This project investigates the use of machine learning for passive beamforming optimization in a Reconfigurable Intelligent Surface (RIS)-assisted wireless communication system.



The system considers a single-input single-output (SISO) communication link in which the signal from the base station reaches the user through an RIS consisting of multiple reflecting elements.



The main objective is to learn a mapping from the cascaded BS-RIS-user channel to discrete RIS phase configurations and evaluate how closely the learned configuration approaches analytical and optimization-based reference methods.



The project combines concepts from:



\- Wireless communications

\- Reconfigurable Intelligent Surfaces (RIS)

\- Rayleigh fading channels

\- Passive beamforming

\- Discrete phase optimization

\- Machine learning

\- Communication system performance evaluation



\---



\## Problem Statement



An RIS can improve wireless communication performance by adjusting the phase shift introduced by each reflecting element.



For an RIS with \\(N\\) elements, the effective cascaded channel can be written as



\\\[

h\_{\\mathrm{eff}}

=

\\sum\_{n=1}^{N}

h\_{\\mathrm{RU},n}

\\phi\_n

h\_{\\mathrm{BR},n}

\\]



where:



\- \\(h\_{\\mathrm{BR},n}\\) is the BS-to-RIS channel for element \\(n\\)

\- \\(h\_{\\mathrm{RU},n}\\) is the RIS-to-user channel for element \\(n\\)

\- \\(\\phi\_n\\) is the phase coefficient of RIS element \\(n\\)



The RIS phase coefficients are constrained to a discrete set in the considered system.



For a 2-bit RIS,



\\\[

\\mathcal{F}

=

\\left\\{

0,\\frac{\\pi}{2},\\pi,\\frac{3\\pi}{2}

\\right\\}.

\\]



The challenge is to select an appropriate phase for every RIS element to obtain a strong effective channel.



\---



\## System Model



The project considers a SISO BS-RIS-user system with:



\- \\(N=16\\) RIS elements

\- Rayleigh fading

\- Quasi-static block fading

\- Narrowband transmission

\- Unit transmit power

\- Unit noise power

\- No direct BS-user link in the initial system model



The BS-RIS and RIS-user channels are modeled as independent complex Gaussian channels:



\\\[

h\_{\\mathrm{BR},n}

\\sim

\\mathcal{CN}(0,1)

\\]



\\\[

h\_{\\mathrm{RU},n}

\\sim

\\mathcal{CN}(0,1).

\\]



The cascaded channel for each RIS element is



\\\[

c\_n

=

h\_{\\mathrm{BR},n}h\_{\\mathrm{RU},n}.

\\]



Therefore,



\\\[

h\_{\\mathrm{eff}}

=

\\sum\_{n=1}^{N} c\_n\\phi\_n.

\\]



The received SNR is



\\\[

\\mathrm{SNR}

=

\\frac{P|h\_{\\mathrm{eff}}|^2}

{\\sigma^2}.

\\]



The achievable rate is evaluated using



\\\[

R

=

\\log\_2(1+\\mathrm{SNR}).

\\]



\---



\## RIS Phase Optimization



\### Continuous-Phase Reference



For continuous RIS phase control, the phase of each element can be selected to compensate for the phase of the cascaded channel:



\\\[

\\theta\_n^\*

=

\-\\angle(c\_n).

\\]



Therefore,



\\\[

\\phi\_n^\*

=

e^{j\\theta\_n^\*}.

\\]



This aligns the contributions from the different RIS elements and provides coherent combining.



\---



\### Quantized Analytical Phase Selection



Real RIS implementations generally use a finite number of phase states.



For the 2-bit RIS used in this project,



\\\[

\\mathcal{F}

=

\\left\\{

0,\\frac{\\pi}{2},\\pi,\\frac{3\\pi}{2}

\\right\\}.

\\]



The continuous optimal phase is quantized to the nearest available phase level.



This provides a simple analytical reference for evaluating the ML model.



\---



\### Coordinate Descent Optimization



A coordinate descent method is also implemented as an optimization-based benchmark.



The algorithm optimizes one RIS element at a time.



For each RIS element, all available discrete phase values are evaluated while keeping the other RIS phases fixed. The phase producing the highest effective channel gain is selected.



The process is repeated for multiple iterations.



This provides a practical heuristic optimization reference without performing exhaustive search.



For \\(N=16\\) and four possible phase states per element, exhaustive search would require evaluating



\\\[

4^{16}

=

4,294,967,296

\\]



possible configurations.



This illustrates why heuristic optimization methods become useful as the number of RIS elements increases.



\---



\## Machine Learning Approach



The machine learning model learns the relationship between the cascaded channel and the corresponding discrete RIS phase configuration.



Instead of using the four separate channel vectors as the input, the final ML formulation directly uses the cascaded channel



\\\[

c\_n=h\_{\\mathrm{BR},n}h\_{\\mathrm{RU},n}.

\\]



The input features consist of the real and imaginary components:



\\\[

\\mathbf{x}

=

\[

\\Re(c\_1),\\ldots,\\Re(c\_N),

\\Im(c\_1),\\ldots,\\Im(c\_N)

].

\\]



For \\(N=16\\), this produces



\\\[

2N=32

\\]



input features.



The target contains one of four phase classes for each RIS element:



\\\[

y\_n

\\in

\\{0,1,2,3\\}.

\\]



The classes correspond to



| Class | RIS Phase |

|---|---:|

| 0 | \\(0\\) |

| 1 | \\(\\pi/2\\) |

| 2 | \\(\\pi\\) |

| 3 | \\(3\\pi/2\\) |



The training targets are generated from the analytical phase-selection rule followed by 2-bit phase quantization.



Therefore, the ML model is designed to \*\*learn an approximation of the analytical phase-selection mapping\*\*, rather than replacing the analytical solution with an unexplained black-box target.



\---



\## Neural Network Architecture



A shared neural network extracts features from the cascaded channel.



The architecture is:



```text

Input: 32 features

&#x20;       |

&#x20;    Linear

&#x20;     256

&#x20;       |

&#x20;     ReLU

&#x20;       |

&#x20;    Linear

&#x20;     128

&#x20;       |

&#x20;     ReLU

&#x20;       |

&#x20;    Linear

&#x20;      64

&#x20;       |

&#x20;     ReLU

&#x20;       |

&#x20;  16 Phase Heads

&#x20;       |

&#x20;  4 Classes / Element

