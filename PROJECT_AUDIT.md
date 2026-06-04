# Project audit: implementation and manuscript completion

This audit uses the `research-paper-writing` paper-review checklist and the PDF workflow expectations for manuscript delivery.  It records the current claim-evidence status of the code and draft.

## Issues fixed in this pass

1. Manuscript naming was inconsistent.  The LaTeX source is now `paper/manuscript.tex`, the bibliography cache is `paper/manuscript.bbl`, and the target compiled artifact is `paper/manuscript.pdf`.
2. The validation script previously mixed point-to-triangle edge/vertex regions with physical sharp normal-sector queries.  `normal_sector_sharp_mask(...)` now counts only competing active normal sectors and excludes smooth benchmark artifacts.
3. The sharp-feature metric is now covered by tests so a smooth ellipsoid benchmark cannot accidentally contribute to normal-cone hit statistics.
4. The manuscript now states that reported validation focuses on gap, normal and multi-sector behavior.  Hessian support remains part of the interface and theory, but Hessian accuracy is not claimed by the current tables.
5. `scripts/build_and_validate.py` no longer rewrites tracked benchmark data when the files already exist, so running validation does not create unrelated data diffs.
6. The validation report, README, highlights and manuscript tables now use the same current metrics.
7. `scripts/generate_paper_figures.py` now regenerates `paper/figures/08_numerical_validation/fig_results.pdf` from `results/validation_summary.json`.
8. Smooth ellipsoid validation now enables Hessian fitting for the feature-adaptive atlas, so Figure 5(b) compares the intended smooth jet model rather than a piecewise-constant normal fallback.
9. The projection backend now exposes welded-edge and welded-vertex incident corner-normal sectors. This fixed the cone-apex undercount without adding cone-specific logic.
10. Added a supplemental validation suite covering sphere exact Hessian, wedge, capped cylinder, cone zones, ablations, mesh consistency, noisy normals, torus and rigid-transform invariance.

## Remaining implementation risks

1. The prototype now exposes welded feature sectors during projection, but it still does not materialize a full global normal-sector graph with persistent chart labels.  That remains the next production-level data structure.
2. Hessian estimates are implemented by finite differences of projected normals.  The sphere benchmark supports the smooth-Hessian claim on clean data, but noisy normals show that derivative estimates need smoothing or quality checks for scanned/imported meshes.
3. The projection baseline is KD-tree candidate based, not a full BVH or exact global closest-feature oracle.  Larger meshes need stronger candidate search validation.
4. No dynamic contact experiment is implemented yet.  The manuscript currently positions dynamic tests as future work, which is accurate but limits submission strength.

## Remaining manuscript risks

1. Author, affiliation, funding, CRediT, competing-interest and AI-use declarations remain placeholders and must be finalized before submission.
2. The paper still presents a general atlas design that is broader than the Python prototype.  The text now narrows unsupported claims, but reviewers may still ask for explicit normal-sector complex construction.
3. The experiments are geometric oracle tests, not full contact solver tests.  A CMAME-strength version should add at least one dynamic benchmark with force, energy, constraint violation and runtime histories.
4. The numerical evidence is now broader, but it is still geometric.  A dynamic benchmark is the main missing item for a CMAME-strength submission package.

## Completion path

1. Add explicit normal-sector graph construction and chart/feature labels to the code.
2. Add denoising or normal-quality controls before Hessian fitting on imported meshes.
3. Add a dynamic contact example, preferably prism edge impact or cone rim/apex contact.
4. Replace all manuscript placeholders with final author and submission metadata.
