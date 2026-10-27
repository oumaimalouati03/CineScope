# Présentation du projet de stage — CineScope

## Intitulé
Conception et développement d'une application web de recommandation personnalisée de films.

## Problématique
Comment exploiter les notes, les genres et les interactions passées des utilisateurs afin de leur proposer des films susceptibles de les intéresser, tout en conservant leur collection et leur historique ?

## Objectif général
Créer un prototype web inspiré des plateformes de découverte de films, reposant sur un moteur hybride de recommandation.

## Objectifs spécifiques
- Préparer et analyser le jeu de données MovieLens.
- Concevoir un catalogue consultable et filtrable.
- Implémenter une recommandation basée sur le contenu et une approche collaborative.
- Prévoir une stratégie de repli pour les nouveaux profils.
- Fournir un score relatif et un motif de recommandation.
- Enregistrer les notes, favoris et consultations dans SQLite.
- Évaluer les recommandations et documenter les limites du prototype.

## Technologies
Python, Pandas, NumPy, scikit-learn, SciPy, Streamlit, SQLite.

## Livrables présents
Code source, dataset exploité, interface web, moteur de recommandation, persistance de l'historique, guide de lancement, Dockerfile et test unitaire de base.

## Planning indicatif sur 8 semaines
1. S1–S2 : analyse du besoin, exploration/nettoyage des données, conception.
2. S3–S4 : recherche, moteur de contenu et baseline de popularité.
3. S5–S6 : filtrage collaboratif, hybridation, profils et historique.
4. S7–S8 : interface, tests, documentation et démonstration.

## Évolutions proposées
API FastAPI documentée, authentification sécurisée, métriques de ranking (Precision@K, Recall@K, NDCG), évaluation hors ligne avec séparation temporelle, PostgreSQL/Redis, Docker et éventuellement un modèle de deep learning après validation d'une baseline.
