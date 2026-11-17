"""Run the final controlled analyses without overwriting existing results."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.entity_regularized_matrix_factorization import (
    EntityRegularizedMatrixFactorization,
)
from src.evaluation import _ranking_values
from src.genre_content import GenreContentModel
from src.matrix_factorization import RegularizedMatrixFactorization
from src.mf_pipeline import _read_locked_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-1m"
LOCKED_SPLIT = ROOT / "results" / "split_assignments.csv"
OUTPUT = ROOT / "results" / "final_analysis"
SEED = 42
BOOTSTRAPS = 1000
K = 10
ALPHAS = {"chronological": 1.0, "random": 0.8}
FUSION_ALPHAS = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))
SPLIT_HASHES = {
    "chronological": "f6f5f45d8d4bb8cdebb5e1559537b4e8d32e1f455c650e0f7ad989e693093538",
    "random": "0446afd053607c6c2a73eb9d37816bb7a500e3a94c4981745d6486cee993bbc9",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _random_split(ratings):
    rng = np.random.default_rng(SEED)
    labels = np.full(len(ratings), "", dtype=object)
    ordered = ratings.sort_values("row_order", kind="mergesort")
    for _, group in ordered.groupby("user_id", sort=True):
        positions = rng.permutation(group.index.to_numpy())
        n = len(positions)
        train_end = math.floor(0.8 * n)
        validation_end = train_end + math.floor(0.1 * n)
        labels[positions[:train_end]] = "train"
        labels[positions[train_end:validation_end]] = "validation"
        labels[positions[validation_end:]] = "test"
    if not np.all(labels != ""):
        raise ValueError("Random split left rows unassigned.")
    split = ratings.copy()
    split["partition"] = labels
    return split.sort_values("row_order", kind="mergesort").reset_index(drop=True)


class MostPopularModel:
    def fit(self, training):
        counts = training.groupby("movie_id").size()
        self.movie_ids = counts.index.to_numpy(dtype=np.int64)
        self.counts = counts.to_numpy(dtype=np.float64)
        return self

    def predict_many(self, user_ids, movie_ids, clip=True):
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        positions = np.searchsorted(self.movie_ids, movie_ids)
        known = positions < len(self.movie_ids)
        scores = np.zeros(len(movie_ids), dtype=np.float64)
        scores[known] = self.counts[positions[known]]
        return scores


def _load_checkpoint(directory, model_type):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = model_type(
        n_factors=config["n_factors"],
        regularization=config["regularization"],
        learning_rate=config["learning_rate"],
        random_seed=config["random_seed"],
        batch_size=config["batch_size"],
    )
    with np.load(directory / "model_checkpoint.npz", allow_pickle=False) as data:
        for name in (
            "user_ids",
            "movie_ids",
            "user_biases",
            "movie_biases",
            "user_factors",
            "movie_factors",
        ):
            setattr(model, name, data[name].copy())
        model.global_mean = float(data["global_mean"])
    return model


def _load_genre(directory):
    model = GenreContentModel()
    with np.load(directory / "model_checkpoint.npz", allow_pickle=False) as data:
        for name in (
            "movie_ids",
            "genre_names",
            "movie_genre_matrix",
            "user_ids",
            "user_profiles",
            "user_history_counts",
            "profile_norms",
        ):
            setattr(model, name, data[name].copy())
        model.global_mean = float(data["global_mean"])
        model.movie_genre_norms = np.linalg.norm(model.movie_genre_matrix, axis=1)
    return model


def _load_models(split_name):
    root = ROOT / "results"
    if split_name == "chronological":
        sensitivity = root / "regularization_sensitivity"
        return {
            "MF A": _load_checkpoint(
                sensitivity / "A_interaction_weighted",
                RegularizedMatrixFactorization,
            ),
            "MF B": _load_checkpoint(
                sensitivity / "B_entity_level",
                EntityRegularizedMatrixFactorization,
            ),
            "Genre": _load_genre(root / "genre_based"),
        }
    sensitivity = root / "randomized_split_seed42" / "matrix_factorization"
    return {
        "MF A": _load_checkpoint(
            sensitivity / "A_interaction_weighted",
            RegularizedMatrixFactorization,
        ),
        "MF B": _load_checkpoint(
            sensitivity / "B_entity_level",
            EntityRegularizedMatrixFactorization,
        ),
        "Genre": _load_genre(
            root / "randomized_split_seed42" / "genre_content"
        ),
    }


def _ranking_rows(model, movies, split, partition, clip=True, score_kind="mf"):
    training = split.loc[split["partition"] == "train"]
    target = split.loc[split["partition"] == partition]
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant = (
        target.loc[target["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    rows = []
    tie_candidates = 0
    tie_top10 = 0
    users_with_ties = 0
    users_with_top_ties = 0
    for user_id in sorted(target["user_id"].unique()):
        candidates = np.asarray(
            [movie for movie in catalogue if movie not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        if len(candidates) < K:
            continue
        users = np.full(len(candidates), int(user_id), dtype=np.int64)
        if score_kind == "genre":
            scores = model.predict_similarity(users, candidates)
        else:
            scores = model.predict_many(users, candidates, clip=clip)
        order = np.lexsort((candidates, -scores))
        ranked = candidates[order]
        top_scores = scores[order[:K]]
        if clip and score_kind == "mf":
            candidate_ties = int(np.count_nonzero(scores == 5.0))
            top_ties = int(np.count_nonzero(top_scores == 5.0))
            tie_candidates += candidate_ties
            tie_top10 += top_ties
            users_with_ties += int(candidate_ties > 1)
            users_with_top_ties += int(top_ties > 1)
        precision, recall, ndcg = _ranking_values(
            ranked, relevant.get(int(user_id), set()), K
        )
        rows.append(
            {
                "user_id": int(user_id),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "precision_at_10": float(frame["precision_at_10"].mean()),
        "recall_at_10": float(frame["recall_at_10"].dropna().mean()),
        "ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
        "users_evaluated": int(len(frame)),
        "users_excluded_zero_relevant": int(frame["recall_at_10"].isna().sum()),
        "candidates_tied_at_5": tie_candidates,
        "top10_slots_tied_at_5": tie_top10,
        "users_with_candidate_ties_at_5": users_with_ties,
        "users_with_top10_ties_at_5": users_with_top_ties,
    }


def _popularity_metrics(movies, split, partition):
    model = MostPopularModel().fit(split.loc[split["partition"] == "train"])
    return _ranking_rows(model, movies, split, partition, score_kind="popularity")[1]


def _hit_pattern(model, movies, split, partition, score_kind="mf"):
    training = split.loc[split["partition"] == "train"]
    target = split.loc[split["partition"] == partition]
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant = (
        target.loc[target["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    first_hit = []
    early_hit_share = []
    relevant_counts = []
    hit_users = 0
    for user_id in sorted(target["user_id"].unique()):
        candidates = np.asarray(
            [movie for movie in catalogue if movie not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        if len(candidates) < K:
            continue
        users = np.full(len(candidates), int(user_id), dtype=np.int64)
        scores = (
            model.predict_similarity(users, candidates)
            if score_kind == "genre"
            else model.predict_many(users, candidates, clip=False)
        )
        order = np.lexsort((candidates, -scores))
        ranked = candidates[order[:K]]
        relevant_ids = relevant.get(int(user_id), set())
        hits = [int(movie_id in relevant_ids) for movie_id in ranked]
        if not hits or not any(hits):
            continue
        hit_users += 1
        first_hit.append(next(index + 1 for index, hit in enumerate(hits) if hit))
        early_hit_share.append(sum(hits[:3]) / sum(hits))
        relevant_counts.append(len(relevant_ids))
    return {
        "users_with_at_least_one_top10_hit": hit_users,
        "mean_rank_first_hit": float(np.mean(first_hit)),
        "share_of_hits_in_positions_1_3": float(np.mean(early_hit_share)),
        "mean_relevant_items_users_hit": float(np.mean(relevant_counts)),
    }


def _save_checkpoint(model, path):
    np.savez_compressed(
        path,
        user_ids=model.user_ids,
        movie_ids=model.movie_ids,
        user_biases=model.user_biases,
        movie_biases=model.movie_biases,
        user_factors=model.user_factors,
        movie_factors=model.movie_factors,
        global_mean=np.asarray(model.global_mean),
    )


def _ranking_selected_models(split_name, split, movies):
    training = split.loc[split["partition"] == "train"].copy()
    validation = split.loc[split["partition"] == "validation"].copy()
    model_rows = []
    selected = {}
    for model_name, model_type in (
        ("MF A", RegularizedMatrixFactorization),
        ("MF B", EntityRegularizedMatrixFactorization),
    ):
        for factors in (10, 20, 50):
            for regularization in (0.02, 0.05, 0.1):
                model = model_type(
                    n_factors=factors,
                    regularization=regularization,
                    learning_rate=0.01,
                    random_seed=SEED,
                ).fit(training, validation, max_epochs=30, patience=4)
                _, metrics = _ranking_rows(
                    model, movies, split, "validation", clip=False
                )
                model_rows.append(
                    {
                        "model": model_name,
                        "factors": factors,
                        "lambda": regularization,
                        "selected_epoch": model.best_epoch_,
                        "validation_rmse": model.best_validation_rmse_,
                        "validation_ndcg_at_10": metrics["ndcg_at_10"],
                        "model_object": model,
                    }
                )
        candidates = [
            row for row in model_rows if row["model"] == model_name
        ]
        chosen = sorted(
            candidates,
            key=lambda row: (
                -row["validation_ndcg_at_10"],
                row["validation_rmse"],
                row["factors"],
                row["lambda"],
            ),
        )[0]
        selected[model_name] = chosen
    output_rows = []
    for row in model_rows:
        output_rows.append(
            {key: value for key, value in row.items() if key != "model_object"}
        )
    output = pd.DataFrame(output_rows)
    selected_output = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "model_object"}
         for row in selected.values()]
    )
    return selected, output, selected_output


def _popularity_hybrid_rows(popularity, genre, movies, split, partition, alpha):
    training = split.loc[split["partition"] == "train"]
    target = split.loc[split["partition"] == partition]
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant = (
        target.loc[target["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    rows = []
    for user_id in sorted(target["user_id"].unique()):
        candidates = np.asarray(
            [movie for movie in catalogue if movie not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        users = np.full(len(candidates), int(user_id), dtype=np.int64)
        pop_scores = popularity.predict_many(users, candidates)
        genre_scores = genre.predict_similarity(users, candidates)
        pop_z = (pop_scores - pop_scores.mean()) / max(pop_scores.std(), 1e-8)
        genre_z = (genre_scores - genre_scores.mean()) / max(
            genre_scores.std(), 1e-8
        )
        order = np.lexsort(
            (candidates, -(alpha * pop_z + (1 - alpha) * genre_z))
        )
        precision, recall, ndcg = _ranking_values(
            candidates[order], relevant.get(int(user_id), set()), K
        )
        rows.append(
            {
                "user_id": int(user_id),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
    return pd.DataFrame(rows)


def _hybrid_rows(mf, genre, movies, split, partition, alpha):
    training = split.loc[split["partition"] == "train"]
    target = split.loc[split["partition"] == partition]
    catalogue = np.sort(movies["movie_id"].to_numpy(dtype=np.int64))
    rated = training.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant = (
        target.loc[target["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    rows = []
    for user_id in sorted(target["user_id"].unique()):
        candidates = np.asarray(
            [movie for movie in catalogue if movie not in rated.get(user_id, set())],
            dtype=np.int64,
        )
        users = np.full(len(candidates), int(user_id), dtype=np.int64)
        mf_scores = mf.predict_many(users, candidates, clip=False)
        genre_scores = genre.predict_similarity(users, candidates)
        mf_z = (mf_scores - mf_scores.mean()) / max(mf_scores.std(), 1e-8)
        genre_z = (genre_scores - genre_scores.mean()) / max(
            genre_scores.std(), 1e-8
        )
        order = np.lexsort((candidates, -(alpha * mf_z + (1 - alpha) * genre_z)))
        precision, recall, ndcg = _ranking_values(
            candidates[order], relevant.get(int(user_id), set()), K
        )
        rows.append(
            {
                "user_id": int(user_id),
                "precision_at_10": precision,
                "recall_at_10": recall,
                "ndcg_at_10": ndcg,
            }
        )
    return pd.DataFrame(rows)


def _select_fusion_alpha(mf, genre, movies, split):
    rows = []
    for alpha in FUSION_ALPHAS:
        frame = _hybrid_rows(mf, genre, movies, split, "validation", alpha)
        rows.append(
            {
                "alpha": alpha,
                "precision_at_10": float(frame["precision_at_10"].mean()),
                "recall_at_10": float(frame["recall_at_10"].dropna().mean()),
                "ndcg_at_10": float(frame["ndcg_at_10"].dropna().mean()),
            }
        )
    table = pd.DataFrame(rows)
    chosen = table.sort_values(
        ["ndcg_at_10", "alpha"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    return float(chosen["alpha"]), table


def _bootstrap(frame_map):
    rng = np.random.default_rng(SEED)
    common_users = None
    for frame in frame_map.values():
        frame_users = set(frame.loc[frame["ndcg_at_10"].notna(), "user_id"])
        common_users = frame_users if common_users is None else common_users & frame_users
    user_ids = np.asarray(sorted(common_users), dtype=np.int64)
    if not len(user_ids):
        raise ValueError("No users have NDCG values for every paired model.")
    index = {
        name: frame.set_index("user_id").loc[user_ids]
        for name, frame in frame_map.items()
    }
    rows = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(user_ids), len(user_ids))
        values = {
            name: frame.iloc[sampled]["ndcg_at_10"].to_numpy()
            for name, frame in index.items()
        }
        rows.append(
            {
                "MF A": float(values["MF A"].mean()),
                "MF B": float(values["MF B"].mean()),
                "Hybrid": float(values["Hybrid"].mean()),
                "MF B - MF A": float((values["MF B"] - values["MF A"]).mean()),
                "Hybrid - MF B": float((values["Hybrid"] - values["MF B"]).mean()),
            }
        )
    samples = pd.DataFrame(rows)
    summary = {}
    for column in samples:
        summary[column] = {
            "estimate": float(samples[column].mean()),
            "ci95_lower": float(samples[column].quantile(0.025)),
            "ci95_upper": float(samples[column].quantile(0.975)),
        }
    return summary, samples


def _paired_bootstrap(frame_map, comparisons):
    rng = np.random.default_rng(SEED)
    common_users = None
    for frame in frame_map.values():
        users = set(frame.loc[frame["ndcg_at_10"].notna(), "user_id"])
        common_users = users if common_users is None else common_users & users
    user_ids = np.asarray(sorted(common_users), dtype=np.int64)
    indexed = {
        name: frame.set_index("user_id").loc[user_ids]
        for name, frame in frame_map.items()
    }
    rows = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(user_ids), len(user_ids))
        values = {
            name: frame.iloc[sample]["ndcg_at_10"].to_numpy()
            for name, frame in indexed.items()
        }
        row = {}
        for label, (left, right) in comparisons.items():
            row[label] = float((values[left] - values[right]).mean())
        rows.append(row)
    samples = pd.DataFrame(rows)
    summary = {
        label: {
            "estimate": float(samples[label].mean()),
            "ci95_lower": float(samples[label].quantile(0.025)),
            "ci95_upper": float(samples[label].quantile(0.975)),
        }
        for label in samples
    }
    return summary, samples


def _controlled_comparison(training, validation, movies):
    rows = []
    model_rows = []
    evaluation_split = pd.concat(
        [
            training.assign(partition="train"),
            validation.assign(partition="validation"),
        ],
        ignore_index=True,
    )
    for formulation, model_type in (
        ("A", RegularizedMatrixFactorization),
        ("B", EntityRegularizedMatrixFactorization),
    ):
        for factors in (10, 20, 50):
            for regularization in (0.02, 0.05, 0.1):
                model = model_type(
                    n_factors=factors,
                    regularization=regularization,
                    learning_rate=0.01,
                    random_seed=SEED,
                ).fit(training, validation, max_epochs=30, patience=4)
                model_rows.append(
                    {
                        "formulation": formulation,
                        "factors": factors,
                        "lambda": regularization,
                        "selected_epoch": model.best_epoch_,
                        "validation_rmse": model.best_validation_rmse_,
                        "model": model,
                    }
                )
    model_frame = pd.DataFrame(model_rows)
    for (formulation, regularization), group in model_frame.groupby(
        ["formulation", "lambda"], sort=True
    ):
        selected = group.sort_values(
            ["validation_rmse", "factors"], kind="mergesort"
        ).iloc[0]
        _, metrics = _ranking_rows(
            selected["model"],
            movies,
            evaluation_split,
            "validation",
            clip=True,
        )
        rows.append(
            {
                "formulation": formulation,
                "factors": int(selected["factors"]),
                "lambda": float(regularization),
                "selected_epoch": int(selected["selected_epoch"]),
                "selection": "validation_rmse",
                "validation_rmse": float(selected["validation_rmse"]),
                "validation_ndcg_at_10": metrics["ndcg_at_10"],
            }
        )
    result = pd.DataFrame(rows)
    result["rmse_rank"] = result.groupby("lambda")["validation_rmse"].rank(
        method="first"
    )
    result["ndcg_rank"] = result.groupby("lambda")[
        "validation_ndcg_at_10"
    ].rank(method="first", ascending=False)
    return result


def _sanity_check():
    checks = {}
    for name, model_type, multiplier in (
        ("A", RegularizedMatrixFactorization, 2.0),
        ("B", EntityRegularizedMatrixFactorization, 1.0),
    ):
        model = model_type(
            n_factors=1, regularization=0.1, learning_rate=0.01, random_seed=SEED
        )
        model.global_mean = 0.0
        model.user_ids = np.array([1])
        model.movie_ids = np.array([1, 2])
        model.user_biases = np.array([0.0])
        model.movie_biases = np.zeros(2)
        model.user_factors = np.array([[1.0]])
        model.movie_factors = np.zeros((2, 1))
        model._batch_update(
            np.array([0, 0]), np.array([0, 1]), np.array([0.0, 0.0])
        )
        expected = 1.0 - 0.01 * 0.1 * multiplier
        checks[name] = {
            "observed_user_factor": float(model.user_factors[0, 0]),
            "expected_user_factor": expected,
            "passed": bool(np.isclose(model.user_factors[0, 0], expected)),
        }
    if not all(item["passed"] for item in checks.values()):
        raise AssertionError(checks)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-controlled",
        action="store_true",
        help="Reuse the existing controlled comparison and rerun only frozen evaluations.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ratings, movies, _ = load_dataset(DATA_DIR)
    chronological = _read_locked_split(LOCKED_SPLIT, ratings)
    random_split = _random_split(ratings)
    splits = {"chronological": chronological, "random": random_split}
    for name, split in splits.items():
        path = LOCKED_SPLIT if name == "chronological" else ROOT / "results" / "randomized_split_seed42" / "split_assignments.csv"
        if _sha256(path) != SPLIT_HASHES[name]:
            raise AssertionError(f"{name} split hash changed")

    if not args.skip_controlled:
        controlled = _controlled_comparison(
            chronological.loc[chronological["partition"] == "train"],
            chronological.loc[chronological["partition"] == "validation"],
            movies,
        )
        controlled.to_csv(OUTPUT / "controlled_ab_validation.csv", index=False)
        sanity = _sanity_check()
        (OUTPUT / "regularization_sanity.json").write_text(
            json.dumps(sanity, indent=2), encoding="utf-8"
        )
    elif not (OUTPUT / "controlled_ab_validation.csv").exists():
        raise FileNotFoundError("Cannot skip controlled analysis before it has run.")

    ranking_summary = {}
    bootstrap_summary = {}
    bootstrap_samples = {}
    ranking_selection = {}
    hit_patterns = {}
    fusion_tables = {}
    for split_name, split in splits.items():
        models = _load_models(split_name)
        popularity = MostPopularModel().fit(
            split.loc[split["partition"] == "train"]
        )
        ranking_selected, all_selected_rows, selected_rows = (
            _ranking_selected_models(split_name, split, movies)
        )
        all_selected_rows.to_csv(
            OUTPUT / f"{split_name}_ndcg_selection_grid.csv", index=False
        )
        selected_rows.to_csv(
            OUTPUT / f"{split_name}_ndcg_selected_configs.csv", index=False
        )
        ranking_selection[split_name] = {}
        ranking_summary[split_name] = {}
        for model_name, model in models.items():
            score_kind = "genre" if model_name == "Genre" else "mf"
            for partition in ("validation", "test"):
                _, metrics = _ranking_rows(
                    model, movies, split, partition, clip=True, score_kind=score_kind
                )
                _, unclipped = _ranking_rows(
                    model, movies, split, partition, clip=False, score_kind=score_kind
                ) if score_kind == "mf" else (None, None)
                ranking_summary[split_name][f"{model_name} {partition}"] = {
                    "clipped": metrics,
                    "unclipped": unclipped,
                }
            if model_name in ("MF A", "MF B"):
                hit_patterns[f"{split_name} {model_name} RMSE"] = _hit_pattern(
                    model, movies, split, "validation"
                )

        popularity_frames = {}
        for partition in ("validation", "test"):
            popularity_frames[partition] = _ranking_rows(
                popularity, movies, split, partition, clip=False, score_kind="popularity"
            )[0]
        ranking_summary[split_name]["Most popular"] = {
            partition: _ranking_rows(
                popularity, movies, split, partition, clip=False, score_kind="popularity"
            )[1]
            for partition in ("validation", "test")
        }
        hit_patterns[f"{split_name} Most popular"] = _hit_pattern(
            popularity, movies, split, "validation", score_kind="popularity"
        )

        selected_models = {}
        for model_name, row in ranking_selected.items():
            selected_models[model_name] = row["model_object"]
            _save_checkpoint(
                row["model_object"],
                OUTPUT / f"{split_name}_{model_name.replace(' ', '_')}_ndcg_checkpoint.npz",
            )
            ranking_selection[split_name][model_name] = {
                key: value for key, value in row.items() if key != "model_object"
            }
            for partition in ("validation", "test"):
                ranking_summary[split_name][
                    f"{model_name} NDCG-selected {partition}"
                ] = _ranking_rows(
                    row["model_object"], movies, split, partition, clip=False
                )[1]
            hit_patterns[f"{split_name} {model_name} NDCG"] = _hit_pattern(
                row["model_object"], movies, split, "validation"
            )

        alpha, fusion_table = _select_fusion_alpha(
            models["MF B"], models["Genre"], movies, split
        )
        fusion_table.to_csv(OUTPUT / f"{split_name}_fusion_validation.csv", index=False)
        fusion_tables[split_name] = {
            "selected_alpha": alpha,
            "validation": fusion_table.to_dict(orient="records"),
        }
        hybrid_frame = _hybrid_rows(
            models["MF B"], models["Genre"], movies, split, "test", alpha
        )
        hybrid_ndcg_frame = _hybrid_rows(
            selected_models["MF B"], models["Genre"], movies, split, "test", alpha
        )
        frame_map = {
            "Popular": popularity_frames["test"],
            "MF A RMSE": _ranking_rows(
                models["MF A"], movies, split, "test", clip=False
            )[0],
            "MF B RMSE": _ranking_rows(
                models["MF B"], movies, split, "test", clip=False
            )[0],
            "MF A NDCG": _ranking_rows(
                selected_models["MF A"], movies, split, "test", clip=False
            )[0],
            "MF B NDCG": _ranking_rows(
                selected_models["MF B"], movies, split, "test", clip=False
            )[0],
            "Hybrid RMSE": hybrid_frame,
            "Hybrid NDCG": hybrid_ndcg_frame,
        }
        comparisons = {
            "Popular - MF B RMSE": ("Popular", "MF B RMSE"),
            "Popular - MF B NDCG": ("Popular", "MF B NDCG"),
            "MF B RMSE - MF A RMSE": ("MF B RMSE", "MF A RMSE"),
            "MF B NDCG - MF A NDCG": ("MF B NDCG", "MF A NDCG"),
        }
        if split_name == "random":
            comparisons.update(
                {
                    "Popular - best hybrid": ("Popular", "Hybrid RMSE"),
                    "Hybrid - MF B RMSE": ("Hybrid RMSE", "MF B RMSE"),
                }
            )
        summary, samples = _paired_bootstrap(frame_map, comparisons)
        bootstrap_summary[split_name] = {
            "alpha_selected_on_validation": alpha,
            "test_ndcg": {
                name: float(frame["ndcg_at_10"].dropna().mean())
                for name, frame in frame_map.items()
            },
            "bootstrap": summary,
        }
        samples.insert(0, "split", split_name)
        bootstrap_samples[split_name] = samples
        ranking_summary[split_name]["Hybrid test"] = {
            "alpha": alpha,
            "metrics": {
                "precision_at_10": float(hybrid_frame["precision_at_10"].mean()),
                "recall_at_10": float(hybrid_frame["recall_at_10"].dropna().mean()),
                "ndcg_at_10": float(hybrid_frame["ndcg_at_10"].dropna().mean()),
            },
        }
        ranking_summary[split_name]["Hybrid NDCG-selected test"] = {
            "alpha": alpha,
            "metrics": {
                "precision_at_10": float(hybrid_ndcg_frame["precision_at_10"].mean()),
                "recall_at_10": float(hybrid_ndcg_frame["recall_at_10"].dropna().mean()),
                "ndcg_at_10": float(hybrid_ndcg_frame["ndcg_at_10"].dropna().mean()),
            },
        }

    (OUTPUT / "ranking_summary.json").write_text(
        json.dumps(ranking_summary, indent=2), encoding="utf-8"
    )
    (OUTPUT / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2), encoding="utf-8"
    )
    pd.concat(bootstrap_samples.values(), ignore_index=True).to_csv(
        OUTPUT / "bootstrap_samples.csv", index=False
    )
    (OUTPUT / "ranking_selection.json").write_text(
        json.dumps(ranking_selection, indent=2), encoding="utf-8"
    )
    (OUTPUT / "hit_patterns.json").write_text(
        json.dumps(hit_patterns, indent=2), encoding="utf-8"
    )
    (OUTPUT / "fusion_selection.json").write_text(
        json.dumps(fusion_tables, indent=2), encoding="utf-8"
    )
    metadata = {
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "test_used_for_selection": False,
        "alpha_selected_on_validation_only": {
            name: value["selected_alpha"]
            for name, value in fusion_tables.items()
        },
        "split_sha256": SPLIT_HASHES,
        "output_files": [
            "controlled_ab_validation.csv",
            "regularization_sanity.json",
            "ranking_summary.json",
            "bootstrap_summary.json",
            "bootstrap_samples.csv",
        ],
    }
    (OUTPUT / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(bootstrap_summary, indent=2))


if __name__ == "__main__":
    main()
