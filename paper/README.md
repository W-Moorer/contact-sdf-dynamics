# Contact-SDF Atlas CMAME draft

This folder contains a complete LaTeX manuscript draft prepared with Elsevier's `elsarticle` class.

Files:
- `manuscript.tex`: manuscript source
- `references.bib`: BibTeX references
- `figures/06_contact_sdf_atlas/`: Contact-SDF Atlas section figures
- `figures/08_numerical_validation/`: Numerical validation section figures
- `figures/08_numerical_validation/zero_level_panels/`: independent zero-level surface panels kept as figure assets
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
