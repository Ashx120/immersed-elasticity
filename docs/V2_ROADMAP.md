# V2: extensions after a verified baseline

V1 is an educational, bilinear immersed elasticity solver. It does not reproduce
the complete high-order thermo-viscoplastic method in the source paper.

## First extension: non-negative moment fitting

- Fit scaled polynomial moments with a non-negative least-squares algorithm.
- Use AST moments with a documented reference-integration tolerance.
- Increase candidate density and fall back to AST if residual criteria fail.
- Verify positivity, area, fitted moments, and independent unfitted functions.
- Compare stress errors and total timings, including moment-fitting setup.
- Report break-even repeated-assembly count rather than equating fewer points
  with faster end-to-end execution.

## Further research features

1. High-order integrated-Legendre basis with p-convergence verification.
2. Hierarchical overlay refinement and gradient-jump error indicators.
3. Linear thermoelasticity with a verified stationary conduction problem.
4. J2 plasticity with material-point tests and a consistent tangent.
5. Rate dependence and temperature-dependent properties.

Each feature needs its own independent verification before combined benchmarks.
Prioritize the first extension; the remaining items are not a short release promise.
