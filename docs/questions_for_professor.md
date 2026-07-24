# Questions for the professor

The forward numerical foundation is verified, but the physical source and
receiver model must be confirmed before comparing it with hardware:

1. What physical quantity is actively prescribed by each transmitter: pressure, normal velocity, displacement, or electrical drive only?
2. What calibrated quantity does each receiver return?
3. Does I/Q represent pressure, particle velocity, or a transfer function relative to a reference channel?
4. Are the source and receiver locations discrete transducers or effectively continuous boundary patterns?
5. Is density assumed known and constant?
6. What attenuation and dispersion model is appropriate?
7. What are the tank geometry, boundary material, frequencies, and transducer bandwidth?
8. Is a water/reference scan subtracted or divided from subject data?
9. Are source phase and receiver phase individually calibrated?
10. Is the intended inverse datum pressure, flux, or a DtN matrix?

Until these are answered, `boundary_flux_response` is explicitly an idealized
Dirichlet-to-Neumann observable, not a claim about calibrated receiver pressure.

