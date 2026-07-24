# Working Problem Definition

## Steady-state modal ultrasound tomography with coherent I/Q measurements

> **Status:** This document is our current, evidence-based interpretation of the
> professor's research direction. It combines the meeting notes, the legacy code,
> and the verified forward simulator. It is not yet a substitute for confirming
> the physical source, receiver, and novelty claims with the professor.

## 1. The idea in one paragraph

The proposed system surrounds a body or tissue sample with acoustic transducers.
Instead of recording only conventional pulse echoes, it drives the boundary with
known sinusoidal patterns, one temporal frequency at a time. The receivers use
coherent I/Q demodulation to retain the amplitude and phase of the steady-state
response. Several spatial boundary patterns and frequencies provide different
views through the object. An inverse algorithm then searches for the internal
sound-speed map whose simulated responses best match the measured responses.

The possible cost-saving hypothesis is:

> Narrowband, coherently driven boundary modes and baseband I/Q measurements may
> retain enough information for quantitative tomography while shifting some
> complexity from broadband acquisition hardware into a calibrated computational
> reconstruction.

That hypothesis is plausible, but it is not proven by the current project.

## 2. What problem are we actually solving?

The primary problem is **quantitative acoustic tomography**:

```text
known acoustic excitations
        +
measured complex responses
        +
a wave-propagation model
        ↓
estimate the internal sound-speed field c(x, y)
```

The desired output is not a photograph and is not literally an MRI image. It is
a two-dimensional map of an acoustic property:

```text
c(x, y) = local speed of sound
```

Different tissues may have different acoustic properties, so a sound-speed map
can reveal internal structure. Determining which structures are medically
meaningful is a later interpretation and validation problem.

### The hierarchy of problems

| Level | Question |
| --- | --- |
| Medical goal | Can the reconstructed acoustic map provide useful soft-tissue information? |
| Physical inverse problem | Which internal `c(x, y)` explains the measured waves? |
| Numerical problem | Can we calculate that map accurately and efficiently? |
| Optimization problem | Can we avoid false minima such as cycle skipping? |
| Acquisition problem | Can a small set of boundary modes capture enough information? |
| Engineering problem | Can the required waves and I/Q data be produced and calibrated inexpensively? |

Cycle skipping is therefore an important obstacle, but it is not the overall
research objective.

## 3. Proposed equipment concept

A possible scanner would contain:

- a circular or cylindrical water tank for acoustic coupling;
- a ring of transmitting and receiving transducers;
- a reference oscillator with a precisely known frequency and phase;
- programmable amplitude and phase control for each transmitting element;
- receiver amplification and synchronized I/Q demodulation;
- low-pass filtering and analog-to-digital conversion;
- temperature monitoring and a reference-water calibration;
- a computer for experiment control and reconstruction.

A simplified acquisition chain is:

```text
reference oscillator at frequency f
                 │
                 ▼
programmable complex drive weights
                 │
                 ▼
ring transducers synthesize boundary pattern g_s(theta)
                 │
                 ▼
waves propagate through the unknown body c(x, y)
                 │
                 ▼
boundary receivers record sinusoidal responses
                 │
                 ▼
synchronized I/Q demodulation and low-pass filtering
                 │
                 ▼
complex response D[receiver, pattern, frequency]
```

This remains a proposed interpretation. Real transducers are electrically
driven and normally return voltages related to acoustic pressure. The current
simulator instead prescribes ideal pressure and observes ideal boundary flux.
The relationship between those two experiments must be established before
working with physical data.

## 4. Why use a steady-state Helmholtz model?

Assume the source emits one sinusoidal angular frequency:

```text
omega = 2 pi f.
```

After transients have decayed, every point in a linear system oscillates at the
same temporal frequency. However, its amplitude and phase depend on position.
We represent the real pressure by

```text
p(x, t) = Re{u(x) exp(-i omega t)}.
```

The complex field

```text
u(x) = I(x) + i Q(x)
```

stores the spatial amplitude and phase. Using the chosen time convention,

```text
p(x, t) = I(x) cos(omega t) + Q(x) sin(omega t).
```

Thus the full time-periodic steady state is represented by one complex value at
each spatial degree of freedom. We do not need to simulate thousands of time
steps after reaching steady state.

### Current physical assumptions

The verified foundation assumes:

- two spatial dimensions;
- scalar acoustic pressure;
- constant density;
- isotropic sound speed;
- a circular domain;
- full boundary access;
- known boundary pressure;
- coherent I/Q measurements;
- a simple explicit loss parameter;
- no transducer impulse response or calibration error;
- no out-of-plane propagation.

These are modeling assumptions, not universal properties of real equipment.

## 5. The forward problem

Define squared slowness and the complex squared wave number:

```text
m(x) = 1 / c(x)^2
k_squared(x) = omega^2 m(x) (1 + i eta),
```

where `eta >= 0` is the configured dimensionless loss.

The current idealized experiment solves

```text
-Laplacian(u) - k_squared(x) u = 0       in Omega,
u = g_s                                  on boundary(Omega).
```

Here:

- `Omega` is the circular imaging region;
- `c(x)` is the supplied sound-speed map;
- `g_s` is the selected boundary forcing pattern;
- `u_s` is the resulting complex pressure field.

The modeled observation is the outward normal derivative:

```text
D_s = du_s/dn       on boundary(Omega).
```

This defines a Dirichlet-to-Neumann map:

```text
Lambda_c(omega): g_s -> D_s.
```

In plain language:

```text
prescribe boundary pressure
        ↓
the medium changes the wave
        ↓
observe the boundary reaction
```

## 6. I/Q demodulation and the “DC” note

A receiver observes a rapidly oscillating signal. Coherent demodulation
multiplies that signal by synchronized cosine and sine references:

```text
received signal × cos(omega t) -> low-pass filter -> I
received signal × sin(omega t) -> low-pass filter -> Q
```

After low-pass filtering, `I` and `Q` are approximately constant baseband, or
“DC,” values during a steady measurement.

The usual derived views are

```text
amplitude = sqrt(I^2 + Q^2)
phase = atan2(Q, I).
```

The reconstruction must retain the original complex pair:

```text
D = I + iQ.
```

Amplitude alone is insufficient because phase contains essential propagation
and travel-time information.

## 7. Boundary forcing patterns

Instead of modeling a single point transmitter, the professor's code uses
spatial Fourier patterns around the circular boundary:

```text
g_0(theta) = 1
g_cos,m(theta) = cos(m theta)
g_sin,m(theta) = sin(m theta)
```

or equivalently

```text
g_exp,m(theta) = exp(i m theta).
```

These are **spatial modes**. They describe how the phase and amplitude of the
drive vary around the ring; they are not additional temporal frequencies.

If a calibrated transducer ring can synthesize these patterns, the measured
data at one frequency have the conceptual shape

```text
D[receiver_angle, boundary_pattern].
```

With multiple frequencies:

```text
D[receiver_angle, boundary_pattern, frequency].
```

The required number and type of boundary patterns remain experimental design
questions. The meeting notes do not establish a data-compression hypothesis.

## 8. The inverse problem

Let `D_observed` be the complex measurements and `D_predicted(c)` be the
responses calculated by the forward simulator for a candidate sound-speed map.
A conventional inverse problem is

```text
minimize over c:

    J(c)
      = 1/2 sum over frequencies and patterns
          ||D_predicted(c) - D_observed||^2
        + alpha R(c),

subject to:

    c_min <= c(x) <= c_max.
```

The first term rewards agreement with measured I/Q data. `R(c)` is a
regularizer or structural prior, and `alpha` controls its strength.

A simple reconstruction loop is:

```text
choose an initial sound-speed map
        ↓
solve every required forward problem
        ↓
compare predicted and observed complex data
        ↓
calculate the objective and its gradient
        ↓
update the sound-speed map
        ↓
repeat
```

The current project implements the forward arrow

```text
c(x) -> D_predicted(c)
```

but not the inverse arrow

```text
D_observed -> estimated c(x).
```

## 9. Why use an adjoint method?

An image may contain thousands or millions of unknown sound-speed values.
Calculating one finite-difference derivative per unknown would require thousands
or millions of additional simulations per iteration.

The adjoint-state method uses the structure of the wave equation to calculate
the complete gradient much more efficiently:

```text
one forward field
        +
one residual-driven adjoint field
        ↓
gradient contribution for the entire sound-speed map
```

The gradient then tells the optimizer how a small change in `c(x)` is expected
to change the data mismatch.

Before trusting an adjoint implementation, its directional derivative must be
verified with Taylor tests.

## 10. Frequency stepping and cycle skipping

Low frequencies have long wavelengths. They are mainly sensitive to large-scale
structure and tolerate larger errors in the initial model. High frequencies add
finer detail but make the objective more oscillatory.

Frequency continuation uses

```text
lowest useful frequency
        ↓
recover a smooth, large-scale model
        ↓
use that result to initialize the next frequency
        ↓
repeat toward higher frequencies and finer detail.
```

Cycle skipping occurs when predicted and observed oscillations are sufficiently
misaligned that the objective associates the predicted wave with the wrong
cycle. The optimizer can then reduce its numerical objective while moving
toward an incorrect sound-speed map.

Frequency stepping attempts to keep the predicted and observed phases close
enough that each new stage begins inside a useful basin of attraction.

Possible later hypotheses include phase-difference or beat-frequency
initialization, alternative objectives, or wavefield-reconstruction methods.
Those methods should be compared only after a conventional FWI control case is
implemented and its cycle-skipping failure is reproducible.

## 11. What might “impose structure” mean?

The meeting note remains ambiguous. Plausible interpretations include:

- smoothness regularization;
- piecewise-smooth or total-variation regularization;
- bounds on physically plausible sound speeds;
- a low-dimensional shape or region parameterization;
- sparsity in a chosen basis;
- anatomical prior information;
- coupling to another imaging modality.

Structure can stabilize an underdetermined acquisition, but it can also
manufacture features that the measurements do not support. Every prior must
therefore be explicit and tested for bias.

The professor should clarify which meaning was intended.

## 12. Why might this be less expensive?

The potential cost argument is a system-level hypothesis:

```text
narrowband steady-state excitation
        +
baseband I/Q detection
        +
computational inversion
        ↓
potentially simpler acquisition hardware
```

Possible advantages include:

- simpler narrowband signal generation;
- lower-rate baseband sampling after demodulation;
- software replacing some hardware complexity;
- non-ionizing acoustic measurements.

Possible disadvantages include:

- precise phase synchronization requirements;
- difficult per-element calibration;
- the need to synthesize accurate spatial pressure patterns;
- long acquisition times if many frequencies are stepped;
- resonances and tank-boundary effects;
- a large computational reconstruction cost;
- reduced resolution at low frequency;
- mismatch between idealized boundary flux and actual receiver voltage.

“Cheaper MRI alternative” should mean potentially similar **useful structural
information**, not the same physical contrast. MRI measures nuclear magnetic
resonance; this system would reconstruct acoustic properties.

## 13. What the current project proves

The foundation presently verifies:

- circular triangular meshes;
- deterministic sound-speed fields;
- complex Helmholtz assembly;
- explicit boundary-pressure enforcement;
- one factorization for multiple boundary modes;
- complex sparse forward solutions;
- weak outward-flux recovery;
- I/Q, magnitude, and phase views;
- analytic Bessel agreement;
- residual, linearity, sign, and mesh-convergence checks;
- phase-preserving serialization and plotting.

It does **not** prove:

- that physical hardware produces the assumed boundary condition;
- that receivers measure the modeled flux;
- that the selected boundary patterns contain enough independent information;
- that an inversion converges;
- that frequency stepping avoids cycle skipping;
- that a reconstruction works with noise or calibration errors;
- that the method works on tissue or in vivo;
- that it provides clinically useful information;
- that the complete system is less expensive.

## 14. Recommended validation path

### Stage 1 — Confirm the physical experiment

Specify:

- transmitter type and geometry;
- electrical drive and acoustic output;
- receiver observable and calibration;
- number and placement of elements;
- available frequency band;
- whether spatial boundary modes can be synthesized;
- tank geometry and boundary material.

### Stage 2 — Define the research hypothesis

Write one falsifiable sentence identifying:

- what is new;
- which existing limitation it addresses;
- why cost or robustness should improve;
- which baseline it must beat.

### Stage 3 — Implement a conventional inverse control

Add:

- an arbitrary sound-speed parameterization;
- a complex-L2 objective;
- regularization;
- an adjoint gradient;
- Taylor-test verification;
- constrained optimization.

### Stage 4 — Avoid an inverse crime

Generate observations:

- on a different mesh;
- with different numerical settings;
- with controlled noise;
- with source and receiver perturbations;
- eventually with a more realistic discrete-transducer model.

### Stage 5 — Freeze the failure case

Create a documented configuration in which conventional FWI cycle-skips. This
becomes the control experiment for evaluating frequency stepping or a new
initialization method.

### Stage 6 — Test the proposed innovation

Measure:

- reconstruction error;
- robustness to starting model and noise;
- number of required patterns and frequencies;
- acquisition time;
- computation time;
- calibration sensitivity;
- estimated hardware complexity and cost.

### Stage 7 — Move to physical phantoms

Progress from:

```text
independent synthetic data
-> water-tank calibration
-> manufactured tissue-mimicking phantom
-> ex-vivo or approved in-vivo validation.
```

A successful same-model synthetic inversion verifies software logic; it does
not establish real-world performance.

## 15. Questions that still need answers

The most important ambiguous phrase from the meeting notes is:

1. What exactly does **impose structure** mean?

Additional questions:

- Is modal boundary excitation the central new acquisition idea?
- Are many transducers driven simultaneously with complex weights?
- Is the measured datum pressure, velocity, flux, or an electrical transfer function?
- Does “DC amplitude” mean both baseband I and Q, or only magnitude?
- Is phase-difference or beat-frequency initialization part of the intended method?
- Which cost dominates the existing system and is expected to decrease?
- What existing scanner and reconstruction algorithm form the baseline?
- Which body part and spatial resolution are targeted?
- What outcome would falsify the central hypothesis?

The detailed hardware questions are maintained in
[`questions_for_professor.md`](questions_for_professor.md).

## 16. Working interpretation of the meeting notes

The notes can be rewritten as the following tentative research program:

> Build a steady-state Helmholtz tomography experiment. Excite a circular
> boundary with known spatial modes at one sinusoidal frequency, and record the
> coherent complex response through I/Q demodulation. Repeat the experiment from
> low to high frequencies. Create synthetic observations, impose an explicit
> structural prior, and use an adjoint-state optimization method to reconstruct
> the sound-speed field. Test whether low-frequency initialization and frequency
> continuation prevent cycle skipping.

This is currently the clearest formal statement of the problem, subject to the
professor's confirmation.

## 17. Related context

These references demonstrate that ring-array ultrasound tomography,
frequency-domain FWI, low-frequency acquisition, and frequency-difference
initialization are active research areas. They provide context and baselines;
they do not establish whether the professor's particular combination is new.

- [Frequency-domain ultrasound waveform tomography using a ring transducer](https://pubmed.ncbi.nlm.nih.gov/26110909/)
- [A low-frequency acquisition device for ultrasound FWI](https://www.sciencedirect.com/science/article/pii/S0301562922004100)
- [Frequency differencing to initialize FWI without cycle skipping](https://pmc.ncbi.nlm.nih.gov/articles/PMC11734264/)
- [Adjoint-state and uncertainty methods in medical-ultrasound FWI](https://pubmed.ncbi.nlm.nih.gov/39170751/)
