# Contact-SDF Atlas CMAME draft

This folder contains a complete LaTeX manuscript draft prepared with Elsevier's `elsarticle` class.

Files:
- `manuscript.tex`: manuscript source
- `references.bib`: BibTeX references
- `fig_pipeline.pdf`, `fig_leaf_types.pdf`, `fig_refinement.pdf`, `fig_results.pdf`, `fig_supplemental_validation.pdf`: figures
- `highlights.txt`: separate highlights draft
- `manuscript.pdf`: compiled PDF

Compile with:

```bash
latexmk -pdf manuscript.tex
```

Regenerate the results figure from the repository root after running validation:

```bash
python scripts/generate_paper_figures.py
```

The author names, affiliations, funding, CRediT, data availability and AI declaration are placeholders and should be edited before any submission.
