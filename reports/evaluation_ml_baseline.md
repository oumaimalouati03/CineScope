# Évaluation du modèle CineScope

## Prédiction des notes
- Train : 80668 évaluations
- Test : 20168 évaluations
- **RMSE : 0.8811**
- **MAE : 0.6686**

Une valeur plus faible de RMSE/MAE indique généralement une meilleure prédiction.

## Classement
- **Precision@10 : 0.1075**
- **Recall@10 : 0.0827**
- Utilisateurs évalués : 610
- Utilisateurs avec au moins un film pertinent évaluable : 599
- Films pertinents éligibles : 9422
- Pertinent signifie une note réelle de test >= 4.0.
- Les films déjà notés à l'entraînement et les films hors catalogue d'entraînement sont exclus du calcul du rappel.

## Exemple de recommandations pour l'utilisateur 1

| Rang | Film | Note prédite |
|---:|---|---:|
| 1 | Taxi Driver (1976) | 5.000 |
| 2 | Hoop Dreams (1994) | 5.000 |
| 3 | Pulp Fiction (1994) | 5.000 |
| 4 | Shawshank Redemption, The (1994) | 5.000 |
| 5 | In the Name of the Father (1993) | 5.000 |
| 6 | Blade Runner (1982) | 5.000 |
| 7 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | 5.000 |
| 8 | Trainspotting (1996) | 5.000 |
| 9 | Godfather, The (1972) | 5.000 |
| 10 | Apartment, The (1960) | 5.000 |
