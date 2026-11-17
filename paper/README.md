# ES 678 paper

`es678_paper.tex` is the two-column, four-page paper source; the supplied
`neurips_2026.sty` course style is unchanged. Compile from this directory with
Tectonic:

```powershell
tectonic es678_paper.tex
```

Follow the repository-root README to reproduce the experiment outputs and
regenerate the numerical figure. `NUMERICAL_AUDIT.md` maps each reported
result to its saved output. Hybrid alpha is selected on validation; each
frozen hybrid is evaluated once on test.
