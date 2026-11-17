# MovieLens Recommendation Lab

A static, client-side showcase for the MovieLens 1M recommendation project.
It compares per-user top-10 lists from the bias baseline, MF A, MF B, the
genre-profile model, and a validation-selected MF B + genre blend. It also
compares the chronological and randomized split results.

## Run locally

From the project root, rebuild the precomputed page data from the saved model
checkpoints and split files:

```powershell
python web_showcase\build_site_data.py
```

This step does not train a model. It scores eligible catalogues for every
user/partition using the saved checkpoints. It writes a small
`web_showcase/data/site_index.json` and deterministic gzip-compressed
recommendation chunks in `web_showcase/data/`. The page loads only the chunk
needed for the selected split, partition, and user. Test-set hybrid
recommendations and metrics are deliberately omitted.

Serve the site over HTTP (the browser fetches its data JSON):

```powershell
cd web_showcase
python -m http.server 8000
```

Open `http://localhost:8000`. Or, while at the project root, run:

```powershell
python -m http.server 8000 --directory web_showcase
```

## Publish on GitHub Pages

The folder is self-contained and has no build framework, backend, package
dependencies, API keys, or externally hosted JavaScript libraries.

1. Put the **contents** of `web_showcase/` at the root of a GitHub repository
   (or in a `docs/` folder if Pages is configured to publish `/docs`).
2. Confirm `index.html`, `styles.css`, `app.js`, `data/site_index.json`, and
   all `data/*.json.gz` chunks are in the published tree.
3. In repository **Settings → Pages**, select the branch and folder containing
   `index.html`.
4. Open the published URL and test both split selectors, validation/test
   selector, user picker, metric selector, and recommendation cards.

No raw ratings file, demographics, model checkpoint, or training source data
is copied into the published site. The JSON contains catalogue titles/genres,
user IDs, top-10 predictions, small per-user aggregate evaluation summaries,
and matching validation/test held-out labels for those recommendations. Publish
only if you are comfortable sharing these derived MovieLens results.

## Reading the comparison

- **Chronological**: train on earlier interactions and evaluate later ones.
  This is the primary future-oriented protocol.
- **Random per-user**: rating rows are shuffled within each user (seed 42) and
  split 80/10/10. Training includes later timestamps for almost every user;
  its higher scores are not a fair gain over the chronological result.
- Candidates exclude training-rated movies. A held-out rating of at least 4
  marks a relevant recommendation. A movie with no held-out rating is
  unjudged, not a confirmed dislike.
- Precision divides by 10. Users with no relevant held-out ratings are
  excluded from aggregate Recall/NDCG.
- Hybrid is evaluated on validation only. The site intentionally omits hybrid
  results from test.
- Score scales differ: MF/bias show predicted ratings, Genre shows cosine
  similarity, and Hybrid shows a standardized fusion score. Compare rankings,
  not the absolute values across model types.

The generated dataset is derived from the saved experiment outputs. Rerunning
the builder is optional when the index and compressed chunks are already
present. Use a current browser with gzip `DecompressionStream` support.
