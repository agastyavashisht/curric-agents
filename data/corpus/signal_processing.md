# Signal Processing — Reference Corpus

## Signals and Systems
A signal is a function that conveys information about a phenomenon. Continuous-time signal: defined for all t ∈ ℝ. Discrete-time signal: defined only at integer sample indices. A system transforms input signals to outputs. Properties: linearity (superposition), time-invariance (shift-invariant response), causality (output depends only on past/present inputs), stability (BIBO).

## Sampling Theorem
Nyquist-Shannon: to perfectly reconstruct a bandlimited signal with maximum frequency fₘₐₓ, sample at fₛ ≥ 2fₘₐₓ. The rate 2fₘₐₓ is the Nyquist rate. Aliasing: if fₛ < 2fₘₐₓ, high-frequency components appear as lower frequencies in the discrete signal — irreversible distortion. Anti-aliasing filter: lowpass filter applied before sampling to remove components above fₛ/2.

## Fourier Series
Any periodic signal with period T can be expressed as a sum of harmonically related sinusoids: `x(t) = Σ cₙ e^(j2πnt/T)`. Coefficients cₙ give the amplitude and phase at each harmonic. Even functions → cosine series only. Odd functions → sine series only. Parseval's theorem: total power = sum of squared magnitudes of Fourier coefficients.

## Fourier Transform
The Fourier Transform (FT) extends Fourier series to aperiodic signals: `X(f) = ∫x(t)e^(-j2πft)dt`. Inverse: `x(t) = ∫X(f)e^(j2πft)df`. Properties: linearity, time-shift ↔ phase shift, time-scaling, convolution in time ↔ multiplication in frequency. A narrow time pulse → broad spectrum; a pure sinusoid → impulse in frequency.

## DFT and FFT
Discrete Fourier Transform (DFT): `X[k] = Σ x[n] e^(-j2πkn/N)`, k=0…N-1. Frequency resolution: Δf = fₛ/N. Complexity O(N²). Fast Fourier Transform (FFT): divide-and-conquer algorithm, complexity O(N log N). `numpy.fft.fft()` computes it. Zero-padding increases frequency resolution appearance but does not add real information.

## Filtering Basics
Ideal lowpass filter passes frequencies below cutoff fc, blocks above. Highpass passes above fc. Bandpass passes a band [f₁, f₂]. Bandstop (notch) blocks a band. Real filters: Butterworth (maximally flat passband), Chebyshev (ripple, steeper rolloff), FIR (finite impulse response, always stable, linear phase), IIR (infinite impulse response, more efficient, may be unstable).

## Convolution
Continuous: `(x*h)(t) = ∫x(τ)h(t-τ)dτ`. Discrete: `y[n] = Σ x[k]h[n-k]`. Convolution in time ≡ multiplication in frequency (convolution theorem). FIR filter output is the convolution of input with the filter's impulse response h[n]. Length of output: N+M-1 for N-point input convolved with M-point filter.

## Windowing
DFT assumes the analyzed segment repeats periodically. If the signal doesn't fit exactly, discontinuities at the edges cause spectral leakage — energy spreading to neighboring frequency bins. Window functions taper the signal to zero at both ends, reducing leakage at the cost of frequency resolution. Rectangular window: no tapering, best resolution, worst leakage. Hamming/Hanning: moderate tradeoff. Blackman: best leakage suppression, widest main lobe.

## STFT and Spectrograms
Short-Time Fourier Transform (STFT): divide signal into overlapping frames, apply window, compute FFT of each frame. `X(t,f)` gives time-varying frequency content. Spectrogram: |STFT|² plotted as heat map (time × frequency × energy). Time-frequency resolution tradeoff: longer window → better frequency resolution, worse time resolution (Heisenberg uncertainty). Hop size controls overlap between frames.

## Noise and SNR
Signal-to-Noise Ratio: SNR = 10 log₁₀(Psignal/Pnoise) in dB. Higher SNR = cleaner signal. Strategies to improve SNR: ensemble averaging (synchronous averaging reduces noise by √N), bandpass filtering (removes out-of-band noise), smoothing, wavelet denoising. Thermal noise (Johnson noise) is white Gaussian: flat power spectral density. Shot noise is Poisson-distributed.
