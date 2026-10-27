
"""Hybrid movie recommender for CineScope.

Classical Machine Learning:
- Content-based filtering with TF-IDF
- Item-based collaborative filtering
- Bayesian popularity fallback
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: pd.DataFrame | None = None,
    ):
        required_movies = {"movieId", "title", "genres"}
        required_ratings = {"userId", "movieId", "rating"}

        if not required_movies.issubset(movies.columns):
            raise ValueError(
                f"Colonnes manquantes dans movies : "
                f"{sorted(required_movies - set(movies.columns))}"
            )
        if not required_ratings.issubset(ratings.columns):
            raise ValueError(
                f"Colonnes manquantes dans ratings : "
                f"{sorted(required_ratings - set(ratings.columns))}"
            )

        self.movies = movies.copy().drop_duplicates("movieId").reset_index(drop=True)
        self.ratings = ratings[["userId", "movieId", "rating"]].copy()

        for col in ("movieId",):
            self.movies[col] = pd.to_numeric(
                self.movies[col], errors="raise"
            ).astype(int)

        for col in ("userId", "movieId"):
            self.ratings[col] = pd.to_numeric(
                self.ratings[col], errors="raise"
            ).astype(int)

        self.ratings["rating"] = pd.to_numeric(
            self.ratings["rating"], errors="raise"
        ).astype(float)

        if self.ratings.empty:
            raise ValueError("Le fichier des évaluations est vide.")

        if self.ratings[["userId", "movieId", "rating"]].isnull().any().any():
            raise ValueError("Le fichier ratings contient des valeurs manquantes.")

        if self.ratings.duplicated(["userId", "movieId"]).any():
            raise ValueError(
                "Plusieurs évaluations existent pour un même utilisateur et film."
            )

        self.movies["title"] = self.movies["title"].fillna("Film sans titre")
        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("")
            .replace("(no genres listed)", "")
        )

        # Préparation des tags.
        if tags is not None and not tags.empty:
            if {"movieId", "tag"}.issubset(tags.columns):
                self.tags = tags[["movieId", "tag"]].copy()
                self.tags["movieId"] = pd.to_numeric(
                    self.tags["movieId"], errors="coerce"
                )
                self.tags = self.tags.dropna(subset=["movieId"])
                self.tags["movieId"] = self.tags["movieId"].astype(int)
                self.tags["tag"] = self.tags["tag"].fillna("").astype(str)
            else:
                self.tags = pd.DataFrame(columns=["movieId", "tag"])
        else:
            self.tags = pd.DataFrame(columns=["movieId", "tag"])

        tag_text = (
            self.tags.groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values))
            .to_dict()
        )

        self.movies["year"] = self.movies["title"].str.extract(
            r"\((\d{4})\)"
        )[0]

        # Les genres ont un poids renforcé dans la représentation du contenu.
        self.movies["features"] = self.movies.apply(
            lambda row: (
                (str(row["genres"]).replace("|", " ") + " ") * 2
                + str(tag_text.get(int(row["movieId"]), ""))
            ).strip(),
            axis=1,
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )
        self.content_matrix = self.vectorizer.fit_transform(
            self.movies["features"]
        )

        self.movie_ids = self.movies["movieId"].to_numpy(dtype=int)
        self.movie_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        # Statistiques et popularité corrigée pour limiter le biais
        # envers les films ayant très peu d'évaluations.
        self.stats = self.ratings.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )

        self.global_mean = float(self.ratings["rating"].mean())
        self.popularity = (
            self.stats["rating_count"]
            / (self.stats["rating_count"] + 20)
            * self.stats["avg_rating"]
            + 20
            / (self.stats["rating_count"] + 20)
            * self.global_mean
        )

        # Filtrage collaboratif item-based.
        # On centre les évaluations par utilisateur pour représenter
        # ses préférences relatives plutôt que ses notes brutes.
        self.user_means = self.ratings.groupby("userId")["rating"].mean()
        self.user_ids = np.sort(self.ratings["userId"].unique())
        self.user_index = {
            int(user_id): index
            for index, user_id in enumerate(self.user_ids)
        }

        row_indices = self.ratings["movieId"].map(self.movie_index)
        col_indices = self.ratings["userId"].map(self.user_index)
        valid = row_indices.notna() & col_indices.notna()

        valid_ratings = self.ratings.loc[valid]
        rows = row_indices.loc[valid].astype(int).to_numpy()
        cols = col_indices.loc[valid].astype(int).to_numpy()

        user_mean_for_rating = valid_ratings["userId"].map(self.user_means)
        centered_values = (
            valid_ratings["rating"].to_numpy(dtype=float)
            - user_mean_for_rating.to_numpy(dtype=float)
        )

        self.item_user_centered = csr_matrix(
            (centered_values, (rows, cols)),
            shape=(len(self.movies), len(self.user_ids)),
        )

        self.collab_similarity = cosine_similarity(
            self.item_user_centered,
            dense_output=False,
        ).tocsr()

        # Les similarités négatives ne servent pas à propager une préférence.
        self.collab_similarity.data = np.maximum(
            self.collab_similarity.data, 0.0
        )
        self.collab_similarity.eliminate_zeros()

    @staticmethod
    def _normalise(series: pd.Series) -> pd.Series:
        """Normalise un score entre 0 et 1."""
        if series.empty:
            return series.astype(float)

        low = float(series.min())
        high = float(series.max())

        if not np.isfinite(low) or not np.isfinite(high):
            return pd.Series(0.0, index=series.index)

        if high - low < 1e-12:
            return pd.Series(0.5, index=series.index)

        return (series - low) / (high - low)

    def recommend(
        self,
        user_id: int | None = None,
        liked_movie_ids: list[int] | None = None,
        limit: int = 12,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """Retourne les meilleurs films non encore évalués."""
        if limit < 1:
            raise ValueError("limit doit être supérieur ou égal à 1.")

        liked = {
            int(movie_id)
            for movie_id in (liked_movie_ids or [])
            if int(movie_id) in self.movie_index
        }

        if user_id is not None:
            user_id = int(user_id)
            user_ratings = self.ratings[
                self.ratings["userId"] == user_id
            ].copy()
        else:
            user_ratings = self.ratings.iloc[0:0].copy()

        known_rated = set(user_ratings["movieId"].astype(int))

        if not user_ratings.empty:
            liked.update(
                user_ratings.loc[
                    user_ratings["rating"] >= 4.0, "movieId"
                ].astype(int)
            )

        movie_index_series = self.movies.set_index("movieId")
        candidate_ids = self.movie_ids

        # 1. Score de contenu : genres et tags des films appréciés.
        content_scores = pd.Series(0.0, index=candidate_ids)

        valid_liked = [
            movie_id for movie_id in liked
            if movie_id in self.movie_index
        ]

        if valid_liked:
            seed_rows = [
                self.movie_index[movie_id]
                for movie_id in valid_liked
            ]

            weights = []
            for movie_id in valid_liked:
                user_movie_rating = user_ratings.loc[
                    user_ratings["movieId"] == movie_id, "rating"
                ]

                if not user_movie_rating.empty:
                    weights.append(
                        max(float(user_movie_rating.iloc[-1]) - 2.5, 0.5)
                    )
                else:
                    weights.append(1.5)

            profile = self.content_matrix[seed_rows].multiply(
                np.asarray(weights, dtype=float)[:, None]
            ).sum(axis=0)

            profile = csr_matrix(profile)
            similarities = cosine_similarity(
                self.content_matrix, profile
            ).ravel()

            content_scores = pd.Series(
                similarities,
                index=self.movie_ids,
            )

        content_scores = self._normalise(content_scores)

        # 2. Score collaboratif : similarités entre films et écarts
        # aux notes moyennes des utilisateurs.
        collaborative_scores = pd.Series(
            0.0, index=candidate_ids
        )

        if user_id is not None and not user_ratings.empty:
            numerator = np.zeros(len(self.movies), dtype=float)
            denominator = np.zeros(len(self.movies), dtype=float)

            user_mean = float(
                self.user_means.get(user_id, self.global_mean)
            )

            for row in user_ratings.itertuples(index=False):
                movie_id = int(row.movieId)

                if movie_id not in self.movie_index:
                    continue

                idx = self.movie_index[movie_id]
                similarities = self.collab_similarity[:, idx].toarray().ravel()

                preference = float(row.rating) - user_mean
                numerator += similarities * preference
                denominator += similarities

            # Retourne un score relatif : 0.5 représente une préférence
            # neutre après normalisation, les scores plus élevés indiquent
            # une préférence estimée plus forte.
            raw_collab = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 1e-12,
            )

            collaborative_scores = pd.Series(
                raw_collab, index=self.movie_ids
            )
            collaborative_scores = self._normalise(
                collaborative_scores
            )

        # 3. Score de popularité avec moyenne bayésienne.
        popularity_scores = self.popularity.reindex(
            candidate_ids
        ).fillna(self.global_mean)

        popularity_scores = self._normalise(popularity_scores)

        # 4. Combinaison hybride.
        has_content = bool(valid_liked)
        has_collaborative = not user_ratings.empty

        if has_content and has_collaborative:
            final_scores = (
                0.45 * content_scores
                + 0.45 * collaborative_scores
                + 0.10 * popularity_scores
            )
        elif has_content:
            final_scores = (
                0.75 * content_scores
                + 0.25 * popularity_scores
            )
        elif has_collaborative:
            final_scores = (
                0.70 * collaborative_scores
                + 0.30 * popularity_scores
            )
        else:
            final_scores = popularity_scores

        result = movie_index_series[
            ["title", "genres", "year"]
        ].copy()

        result["score"] = final_scores.reindex(
            result.index
        ).fillna(0.0)

        result["avg_rating"] = self.stats["avg_rating"].reindex(
            result.index
        )
        result["rating_count"] = self.stats["rating_count"].reindex(
            result.index
        ).fillna(0).astype(int)

        if has_content and has_collaborative:
            reason = "Basé sur vos goûts et vos évaluations"
        elif has_content:
            reason = "Films similaires à votre sélection"
        elif has_collaborative:
            reason = "Basé sur vos évaluations et la popularité"
        else:
            reason = "Populaire auprès des utilisateurs"

        result["reason"] = reason

        if exclude_rated:
            result = result.loc[
                ~result.index.isin(known_rated | liked)
            ]

        result = result.sort_values(
            ["score", "rating_count", "avg_rating"],
            ascending=[False, False, False],
            na_position="last",
        )

        return result.head(limit).reset_index()

    def search(
        self,
        query: str,
        genre: str = "Tous",
        limit: int = 50,
    ) -> pd.DataFrame:
        """Recherche un film par titre et/ou genre."""
        df = self.movies[
            ["movieId", "title", "genres", "year"]
        ].copy()

        if query and query.strip():
            pattern = re.escape(query.strip())
            df = df[
                df["title"].str.contains(
                    pattern, case=False, regex=True, na=False
                )
            ]

        if genre != "Tous":
            df = df[
                df["genres"].str.contains(
                    re.escape(genre),
                    case=False,
                    regex=True,
                    na=False,
                )
            ]

        df = df.merge(
            self.stats.reset_index(),
            on="movieId",
            how="left",
        )

        df["avg_rating"] = df["avg_rating"].fillna(0.0)
        df["rating_count"] = df["rating_count"].fillna(0).astype(int)

        return df.sort_values(
            ["rating_count", "avg_rating"],
            ascending=[False, False],
        ).head(limit).reset_index(drop=True)

    def movie_details(self, movie_id: int) -> dict:
        """Retourne les informations et statistiques d'un film."""
        matches = self.movies.loc[
            self.movies["movieId"] == int(movie_id)
        ]

        if matches.empty:
            raise ValueError(f"Film introuvable : movieId={movie_id}")

        row = matches.iloc[0].to_dict()
        stat = self.stats.loc[int(movie_id)] if int(movie_id) in self.stats.index else None

        row["avg_rating"] = (
            float(stat["avg_rating"]) if stat is not None else 0.0
        )
        row["rating_count"] = (
            int(stat["rating_count"]) if stat is not None else 0
        )

        return row
