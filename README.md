# Beyond Genres

This repository contains the code, derived results, paper source, and static
web showcase for the ES 678 MovieLens 1M recommendation project.

## Quick start

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python reproduce.py
```

`reproduce.py` is the master entry point. It reuses existing result folders by
default, which makes a fresh clone immediately inspectable. To remove generated
results and rerun every stage from scratch:

```powershell
python reproduce.py --force
```

The complete run trains the baseline, MF formulations, genre model, validation
fusion, random-split comparison, final audits, paper figure, and web data.
It uses seed 42 and the locked chronological split. Training can take
substantial time and requires the MovieLens 1M files in `data/ml-1m/`.

Run the unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## Repository contents

- `src/`: data loading, splitting, models, evaluation, and pipelines.
- `main.py`, `mf_main.py`, `sensitivity_main.py`, `genre_main.py`: experiment
  entry points.
- `Scripts/`: final audit and selection-resolution analyses.
- `results/`: locked split, checkpoints, metrics, tables, and figures.
- `paper/`: LaTeX source, style file, paper PDF, and figures.
- `web_showcase/`: self-contained static results website and its precomputed
  data.
- `data/ml-1m/README`: upstream dataset license and citation.

## Data access and citation

MovieLens 1M is not redistributed in this repository because its upstream
license prohibits redistribution without permission. `download_data.py` fetches
the official archive from GroupLens and verifies the expected files. Cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. ACM Transactions on Interactive Intelligent Systems
> 5(4), Article 19. https://doi.org/10.1145/2827872

The derived split and result files may be inspected or regenerated after the
dataset is downloaded. Do not commit credentials, private data, or local
virtual environments.

## Web showcase

Serve the existing showcase from the repository root:

```powershell
python -m http.server 8000 --directory web_showcase
```

Open `http://localhost:8000`. The site is static and has no API keys or
backend. It intentionally contains derived recommendations rather than raw
ratings.



See `AI_DISCLOSURE.md` 
