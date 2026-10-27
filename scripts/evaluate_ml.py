"""Évaluation du modèle CineScope. Lancer depuis la racine : py scripts/evaluate_ml.py"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from ml_baseline import ItemBasedCollaborativeFiltering


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}. Vérifie le dossier data/.")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} : colonnes manquantes {sorted(missing)}")
    return df


def ranking_metrics(model, test: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> dict:
    """Precision@K moyenne sur tous les utilisateurs; Recall@K sur ceux ayant des pertinents."""
    catalog = set(map(int, model.movie_ids))
    precisions, recalls = [], []
    users_with_relevant = 0
    total_hits = total_relevant = 0

    for uid, group in test.groupby("userId"):
        uid = int(uid)
        seen = model.user_rated.get(uid, set())
        relevant = set(group.loc[group["rating"] >= threshold, "movieId"].astype(int))
        # Les films hors catalogue ou déjà notés à l'entraînement ne sont pas éligibles.
        relevant = relevant.intersection(catalog).difference(seen)
        recs = model.recommend(uid, k=k)
        recommended = set(recs["movieId"].astype(int))
        hits = len(recommended.intersection(relevant))
        precisions.append(hits / k)
        total_hits += hits
        total_relevant += len(relevant)
        if relevant:
            recalls.append(hits / len(relevant))
            users_with_relevant += 1

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        "users_evaluated": int(len(precisions)),
        "users_with_relevant_test_items": int(users_with_relevant),
        "total_hits_at_k": int(total_hits),
        "total_relevant_test_items_in_catalog": int(total_relevant),
        "ranking_k": int(k),
        "relevance_threshold": float(threshold),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = read_csv(DATA_DIR / "ratings_train.csv", {"userId", "movieId", "rating"})
    test = read_csv(DATA_DIR / "ratings_test.csv", {"userId", "movieId", "rating"})
    movies = read_csv(DATA_DIR / "movies_clean.csv", {"movieId", "title", "genres"})

    for df in (train, test):
        df["userId"] = pd.to_numeric(df["userId"], errors="raise").astype(int)
        df["movieId"] = pd.to_numeric(df["movieId"], errors="raise").astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="raise").astype(float)
        if df[["userId", "movieId", "rating"]].isnull().any().any():
            raise ValueError("Un CSV contient des valeurs manquantes.")
        if df.duplicated(["userId", "movieId"]).any():
            raise ValueError("Un CSV contient des couples userId/movieId dupliqués.")

    train_pairs = set(zip(train.userId, train.movieId))
    test_pairs = set(zip(test.userId, test.movieId))
    overlap = train_pairs.intersection(test_pairs)
    if overlap:
        raise ValueError(f"Fuite train/test : {len(overlap)} couples sont présents dans les deux fichiers.")

    print(f"Train : {len(train):,} évaluations".replace(",", " "))
    print(f"Test  : {len(test):,} évaluations".replace(",", " "))
    print("Entraînement du modèle item-based...")
    model = ItemBasedCollaborativeFiltering(train, n_neighbors=50, shrinkage=5.0)

    print("Calcul des prédictions...")
    y_true = test["rating"].to_numpy(float)
    y_pred = np.array([model.predict(int(u), int(m))
                       for u, m in zip(test["userId"], test["movieId"])])
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    print("Calcul de Precision@10 et Recall@10...")
    ranking = ranking_metrics(model, test, k=10, threshold=4.0)
    example_user = int(test["userId"].iloc[0])
    example_recs = model.recommend(example_user, k=10, movies=movies)

    results = {
        "model": "ItemBasedCollaborativeFiltering",
        "approach": "Machine Learning classique - filtrage collaboratif item-based",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_users": int(train.userId.nunique()),
        "train_movies": int(train.movieId.nunique()),
        "rmse": rmse, "mae": mae, **ranking,
        "example_user_id": example_user,
        "example_recommendations": example_recs.to_dict(orient="records"),
    }
    json_path = REPORTS_DIR / "evaluation_ml_baseline.json"
    md_path = REPORTS_DIR / "evaluation_ml_baseline.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Évaluation du modèle CineScope", "",
        "## Prédiction des notes",
        f"- Train : {len(train)} évaluations",
        f"- Test : {len(test)} évaluations",
        f"- **RMSE : {rmse:.4f}**",
        f"- **MAE : {mae:.4f}**", "",
        "Une valeur plus faible de RMSE/MAE indique généralement une meilleure prédiction.", "",
        "## Classement",
        f"- **Precision@10 : {ranking['precision_at_10']:.4f}**",
        f"- **Recall@10 : {ranking['recall_at_10']:.4f}**",
        f"- Utilisateurs évalués : {ranking['users_evaluated']}",
        f"- Utilisateurs avec au moins un film pertinent évaluable : {ranking['users_with_relevant_test_items']}",
        f"- Films pertinents éligibles : {ranking['total_relevant_test_items_in_catalog']}",
        "- Pertinent signifie une note réelle de test >= 4.0.",
        "- Les films déjà notés à l'entraînement et les films hors catalogue d'entraînement sont exclus du calcul du rappel.", "",
        f"## Exemple de recommandations pour l'utilisateur {example_user}", "",
        "| Rang | Film | Note prédite |", "|---:|---|---:|",
    ]
    for rank, row in enumerate(example_recs.to_dict(orient="records"), 1):
        title = str(row.get("title", row["movieId"])).replace("|", "\\|")
        lines.append(f"| {rank} | {title} | {row['predicted_rating']:.3f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== Résultats ===")
    print(f"RMSE         : {rmse:.4f}")
    print(f"MAE          : {mae:.4f}")
    print(f"Precision@10 : {ranking['precision_at_10']:.4f}")
    print(f"Recall@10    : {ranking['recall_at_10']:.4f}")
    print(f"Utilisateurs évalués : {ranking['users_evaluated']}")
    print(f"Films pertinents éligibles : {ranking['total_relevant_test_items_in_catalog']}")
    print(f"Rapport JSON : {json_path}")
    print(f"Rapport Markdown : {md_path}")
    print("\nExemple de recommandations :")
    print(example_recs.to_string(index=False))


if __name__ == "__main__":
    main()
