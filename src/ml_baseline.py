
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemBasedCollaborativeFiltering:
    def __init__(
        self,
        ratings: pd.DataFrame,
        n_neighbors: int = 50,
        shrinkage: float = 5.0,
    ):
        required = {"userId", "movieId", "rating"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

        self.ratings = ratings[["userId", "movieId", "rating"]].copy()

        for col in ("userId", "movieId"):
            self.ratings[col] = pd.to_numeric(
                self.ratings[col], errors="raise"
            ).astype(int)

        self.ratings["rating"] = pd.to_numeric(
            self.ratings["rating"], errors="raise"
        ).astype(float)

        if self.ratings.empty:
            raise ValueError("Le jeu d'entraînement est vide.")

        if self.ratings.isnull().any().any():
            raise ValueError("Le jeu contient des valeurs manquantes.")

        if self.ratings.duplicated(["userId", "movieId"]).any():
            raise ValueError("Des couples userId/movieId sont dupliqués.")

        if not np.isfinite(self.ratings["rating"]).all():
            raise ValueError(
                "Les évaluations contiennent des valeurs non finies."
            )

        if not self.ratings["rating"].between(0.5, 5.0).all():
            raise ValueError(
                "Les notes doivent être comprises entre 0.5 et 5."
            )

        if shrinkage < 0:
            raise ValueError("shrinkage doit être positif ou nul.")

        self.shrinkage = float(shrinkage)

        self.user_ids = np.sort(self.ratings["userId"].unique())
        self.movie_ids = np.sort(self.ratings["movieId"].unique())

        self.user_index = {
            int(value): index
            for index, value in enumerate(self.user_ids)
        }
        self.movie_index = {
            int(value): index
            for index, value in enumerate(self.movie_ids)
        }

        self.global_mean = float(self.ratings["rating"].mean())

        # Moyennes des utilisateurs avec lissage.
        user_stats = self.ratings.groupby("userId")["rating"].agg(
            ["mean", "count"]
        )

        user_means = user_stats["mean"].reindex(
            self.user_ids
        ).to_numpy(dtype=float)

        user_counts = user_stats["count"].reindex(
            self.user_ids
        ).to_numpy(dtype=float)

        self.user_means = (
            user_counts * user_means
            + self.shrinkage * self.global_mean
        ) / np.maximum(user_counts + self.shrinkage, 1e-12)

        # Construction de la matrice films x utilisateurs.
        rows = self.ratings["movieId"].map(
            self.movie_index
        ).to_numpy()

        cols = self.ratings["userId"].map(
            self.user_index
        ).to_numpy()

        vals = self.ratings["rating"].to_numpy(dtype=float)

        self.rating_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.movie_ids), len(self.user_ids)),
        )

        # Centrage des notes par rapport à la moyenne utilisateur.
        user_mean_mapping = {
            int(user_id): float(self.user_means[index])
            for index, user_id in enumerate(self.user_ids)
        }

        user_mean_for_rows = self.ratings["userId"].map(
            user_mean_mapping
        ).to_numpy(dtype=float)

        centered_vals = vals - user_mean_for_rows

        self.movie_user = csr_matrix(
            (centered_vals, (rows, cols)),
            shape=(len(self.movie_ids), len(self.user_ids)),
        )

        # Moyennes des films avec lissage bayésien.
        movie_stats = self.ratings.groupby("movieId")["rating"].agg(
            ["mean", "count"]
        ).reindex(self.movie_ids)

        counts = movie_stats["count"].to_numpy(dtype=float)
        means = movie_stats["mean"].to_numpy(dtype=float)

        self.movie_means = (
            counts * means + self.shrinkage * self.global_mean
        ) / np.maximum(counts + self.shrinkage, 1e-12)

        # Calcul des similarités cosinus entre les films.
        n = min(
            max(1, int(n_neighbors)) + 1,
            len(self.movie_ids),
        )

        nn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=n,
            n_jobs=-1,
        )

        nn.fit(self.movie_user)
        distances, indices = nn.kneighbors(self.movie_user)

        sim_rows = []
        sim_cols = []
        sim_values = []

        for i, (distances_row, indices_row) in enumerate(
            zip(distances, indices)
        ):
            for distance, neighbor_index in zip(
                distances_row, indices_row
            ):
                j = int(neighbor_index)

                if i == j:
                    continue

                similarity = 1.0 - float(distance)

                # Garder uniquement les similarités positives.
                if similarity > 1e-12:
                    sim_rows.append(i)
                    sim_cols.append(j)
                    sim_values.append(similarity)

        self.similarity = csr_matrix(
            (sim_values, (sim_rows, sim_cols)),
            shape=(len(self.movie_ids), len(self.movie_ids)),
        )

        # Films déjà évalués par chaque utilisateur.
        self.user_rated = {
            int(user_id): set(group["movieId"].astype(int))
            for user_id, group in self.ratings.groupby("userId")
        }

        # Matrice utilisateurs x films.
        self._rating_matrix = self.rating_matrix.T.tocsr()

    def predict_user(self, user_id: int) -> np.ndarray:
        """Prédit une note pour chaque film du catalogue d'entraînement."""
        user_id = int(user_id)

        if user_id not in self.user_index:
            return self.movie_means.copy()

        user_idx = self.user_index[user_id]
        row = self._rating_matrix.getrow(user_idx)

        rated = np.zeros(len(self.movie_ids), dtype=float)
        rated[row.indices] = 1.0

        user_mean = float(self.user_means[user_idx])

        # Estimation de base personnalisée.
        baseline = user_mean + self.movie_means - self.global_mean

        # Écarts entre les notes de l'utilisateur et sa moyenne.
        deviations = np.zeros(len(self.movie_ids), dtype=float)
        deviations[row.indices] = row.data - user_mean

        # Contribution des films similaires déjà évalués.
        numerator = self.similarity.dot(deviations)
        denominator = self.similarity.dot(rated)

        neighbor_deviation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )

        # Réduire les écarts extrêmes dus aux voisins similaires.
        neighbor_deviation = np.clip(
            neighbor_deviation,
            -1.0,
            1.0,
        )

        # Fiabilité de la contribution des voisins.
        confidence = denominator / (
            denominator + self.shrinkage + 1e-12
        )

        predictions = baseline + confidence * neighbor_deviation

        # Respecter l'échelle des notes : 0.5 à 5.
        return np.clip(predictions, 0.5, 5.0)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Prédit une note; film inconnu : moyenne globale."""
        movie_id = int(movie_id)

        if movie_id not in self.movie_index:
            return self.global_mean

        scores = self.predict_user(int(user_id))
        return float(scores[self.movie_index[movie_id]])

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        movies: pd.DataFrame | None = None,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Renvoie les k meilleurs films non encore notés."""
        if k < 1:
            raise ValueError("k doit être supérieur ou égal à 1.")

        user_id = int(user_id)
        scores = self.predict_user(user_id)

        result = pd.DataFrame({
            "movieId": self.movie_ids.astype(int),
            "predicted_rating": scores,
        })

        if exclude_seen:
            seen = self.user_rated.get(user_id, set())
            result = result[~result["movieId"].isin(seen)]

        if movies is not None:
            cols = ["movieId"] + [
                col
                for col in ("title", "genres")
                if col in movies.columns
            ]

            result = result.merge(
                movies[cols].drop_duplicates("movieId"),
                on="movieId",
                how="left",
            )

        result = result.sort_values(
            ["predicted_rating", "movieId"],
            ascending=[False, True],
        )

        return result.head(k).reset_index(drop=True)
