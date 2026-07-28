# System Concept

## The idea

Surround a target with a ring of acoustic transducers. Drive the boundary with
spatial Fourier modes — cos(mθ) and sin(mθ) — one sinusoidal frequency at a
time. Measure the complex steady-state response on the boundary. Step from low
frequency to high frequency. Reconstruct the internal sound-speed field c(x,y)
by finding the c(x) whose simulated responses match the measured ones.

## What does not change

1. **Boundary Fourier modes.** The excitation patterns are cos(mθ) and sin(mθ).

2. **One frequency at a time.** The system operates at a single temporal
   frequency. The entire domain vibrates at that frequency with spatially
   varying amplitude and phase.

3. **Complex response.** The measurement retains both amplitude and phase:
   D = I + iQ. Neither component is discarded.

4. **Frequency stepping.** Start at low frequency, reconstruct, use the result
   as the starting point for the next higher frequency. Low frequencies see
   large-scale structure. High frequencies add detail.

5. **The forward model.** At each frequency, the physics is the Helmholtz
   equation on a disk with prescribed boundary pressure and measured boundary
   flux:

       −∇²u − (ω²/c(x)²) u = 0       in the disk
       u = g(θ)                        on the boundary
       D = ∂u/∂n                       on the boundary

6. **The inverse problem.** Find c(x) such that the predicted D matches the
   observed D across all modes and frequencies.
