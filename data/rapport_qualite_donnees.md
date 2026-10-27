# Rapport de nettoyage des données — CineScope

## Source
MovieLens ml-latest-small. Les fichiers sources à la racine de ce dossier n'ont pas été modifiés.

## Résultats

| Fichier nettoyé | Lignes | Doublons exacts après nettoyage |
|---|---:|---:|
| movies_clean.csv | 9 742 | 0 |
| ratings_clean.csv | 100 836 | 0 |
| tags_clean.csv | 3 683 | 0 |
| links_clean.csv | 9 742 | 0 |

## Contrôles

- Évaluations : notes comprises entre 0.5 et 5.0, par pas de 0,5.
- Identifiants de films des évaluations sans correspondance : 0.
- Valeurs `tmdbId` manquantes conservées : 8 (pas d'invention de valeurs).
- Titres sans année de sortie extractible depuis le titre : 13; les titres sont conservés, car l'absence d'année dans le titre n'est pas une erreur suffisante pour supprimer un film.
- Les tags distincts d'un même couple utilisateur-film sont conservés : plusieurs tags sont autorisés.

## Découpage pour l'évaluation

- Entraînement : 80,668 lignes.
- Test : 20,168 lignes.
- Proportion test : 20 %, graine aléatoire 42, séparation stratifiée par utilisateur.
- Utilisateurs présents dans train/test : 610 / 610.
- Couples (userId, movieId) partagés entre train et test : 0.

## Fichiers produits

- `movies_clean.csv`, `ratings_clean.csv`, `tags_clean.csv`, `links_clean.csv` : versions nettoyées.
- `ratings_train.csv`, `ratings_test.csv` : découpage pour entraîner et évaluer les modèles.
- `quality_report.json` : rapport détaillé exploitable en Python.

## Important

Ce nettoyage est un prétraitement initial. Avant d'évaluer un moteur de recommandation, il faudra s'assurer que l'entraînement utilise uniquement `ratings_train.csv` et que les performances sont calculées sur `ratings_test.csv`, sans fuite de données.
Le fichier de notes propre conserve les colonnes originales `userId`, `movieId`, `rating`, `timestamp` afin de rester compatible avec le moteur existant.
