# Supplemental validation report

Generated with:

```bash
python scripts/supplemental_validation.py
```

The suite uses the same generic projection, uniform-atlas and adaptive-atlas backends for every case. No case-specific runtime logic is used. The final advantage audit reports:

```text
issues: []
```

## Coverage

| group | cases | main check |
| --- | --- | --- |
| Exact smooth reference | sphere exact SDF/normal/Hessian | adaptive smooth jet improves normal and Hessian error |
| Sharp analytic geometry | two-plane wedge | feature records preserve competing plane normals |
| Mixed smooth/sharp geometry | capped cylinder; cylinder side/rim zones | feature refinement improves rim normal-cone hit rate |
| Cone singular geometry | cone side/rim/apex zones | feature refinement improves side, rim and apex behavior without cone-specific code |
| Ablations | sphere resolution; cylinder feature depth; wedge sector angle | expected trends remain stable under parameter changes |
| Mesh consistency | sphere and cylinder mesh refinements | adaptive behavior is stable across tessellation levels |
| Robustness | noisy sphere normals | noisy input is reported as a limitation case, not counted as a theory advantage check |
| Non-convex smooth geometry | torus | adaptive smooth jets work on concave smooth geometry |
| Invariance | rigid transform | query results are unchanged under rigid-frame mapping |

## Key metrics

| case | uniform | adaptive | conclusion |
| --- | ---: | ---: | --- |
| sphere normal mean (deg) | 1.327 | 0.527 | adaptive smooth jet improves normal accuracy |
| sphere Hessian RMSE | 0.527 | 0.219 | adaptive Hessian fitting improves curvature quantity |
| wedge cone-hit | 1.000 | 1.000 | both exact; adaptive gap is lower |
| capped-cylinder cone-hit | 0.543 | 0.914 | adaptive feature leaves preserve rim sectors |
| cone side normal mean (deg) | 7.997 | 3.822 | adaptive improves side normals |
| cone rim cone-hit | 0.662 | 0.926 | adaptive improves rim sectors |
| cone apex cone-hit | 0.246 | 0.492 | apex improves after welded-vertex sector exposure |
| torus normal mean (deg) | 14.055 | 1.383 | adaptive smooth jets improve non-convex smooth normals |

Rigid-transform invariance error was at floating-point roundoff:

```text
max_phi_abs_diff = 2.50e-16
max_world_normal_abs_diff = 2.22e-16
```

Noisy-normal cases are intentionally retained as stress tests. At 5-10 degrees of input normal noise, adaptive Hessian estimates degrade because finite-difference normal derivatives amplify noise; this is a data-quality limitation, not a specialized-case backend failure.
