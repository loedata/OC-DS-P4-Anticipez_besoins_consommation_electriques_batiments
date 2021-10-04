# OC-DS-P4-Anticipez_besoins_consommation_electriques_batiments
Formation OpenClassrooms - Parcours data scientist - Projet n°4 - Anticipez les besoins en consommation électrique de bâtiments

## Prévision des besoins électriques et des émissions de CO2 de bâtiments 

La ville de Seattle souhaite atteindre son objectif de ville neutre en émissions de carbone en 2050, et s’intéresse de près aux émissions des bâtiments non destinés à l’habitation.

![P4_Seattle](https://user-images.githubusercontent.com/71518818/135107824-5ce0b80d-7a06-4398-afc4-fbbca374072c.png)

## Source
Les données sont accessibles : 
- [Jeu de données](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv)
- [Description des variables 2015](https://data.seattle.gov/dataset/2015-Building-Energy-Benchmarking/h7rm-fz6m)
- [Description des variables 2016](https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy)

## Boîte à outils

![P5_outils](https://user-images.githubusercontent.com/71518818/135888118-2d18a911-8f9a-4537-896e-49320b52443c.png)

## Problématique de la ville de Seattle
Des relevés minutieux ont été effectués en 2015 et en 2016. Cependant, ces relevés sont coûteux à obtenir, et à partir de ceux déjà réalisés, la mission consiste à prédire les émissions de CO2 et la consommation totale d’énergie de bâtiments pour lesquels elles n’ont pas encore été mesurées.

La prédiction se basera sur les données déclaratives du permis d'exploitation commerciale (taille et usage des bâtiments, mention de travaux récents, date de construction..)

Une évaluation de l’intérêt de l’"**[Energy Star Score](https://www.energystar.gov/buildings/facility-owners-and-managers/existing-buildings/use-portfolio-manager/interpret-your-results/what)**" pour la prédiction d’émissions, qui est fastidieux à calculer, est également souhaitée.

## Objectifs

- Réaliser une courte analyse exploratoire.
- Tester différents modèles de prédiction afin de répondre au mieux à la problématique.
- Mettre en place une évaluation rigoureuse des performances de la régression, et optimiser les hyperparamètres et le choix d’algorithme de ML à l’aide d’une validation croisée.

## Erreurs à éviter :

- L’objectif est de te passer des relevés de consommation annuels (attention à la fuite de données), mais rien n'interdit d’en déduire des variables plus simples (nature et proportions des sources d’énergie utilisées). 

- Attention au traitement des différentes variables, à la fois pour trouver de nouvelles informations (peut-on déduire des choses intéressantes d’une simple adresse ?) et optimiser les performances en appliquant des transformations simples aux variables (normalisation, passage au log, etc.).

## Démarche

1. **Préparation des données** pour les rendre applicables aux modèles de machine learning :
    - sépartion des données en entrées (matrice X) et de la variable cible (vecteur y).
    - split du jeu de données en jeu d'entraînement (train) et jeu de test (test).
    - préparation des variables quantitatives : standardisation?
    - préparation des variables qualitatives : encodage / standardisation?
![P4_Preproc](https://user-images.githubusercontent.com/71518818/135103853-6112442d-fba6-406f-a3a1-b98a6efe6614.png)
    
2. **Modélisation** :
    - Essayer plusieurs modèles en utilisant les hyperparamètres de base sur la variable cible transformée en log et sur la variable vible non transformée.
        - conclure de la performance des modèles, la variable cible à utiliser : transformée ou non?
        - sélectionner les 3 modèles avec les hyperparamètres de base les plus performant.
![P4_Modelisation](https://user-images.githubusercontent.com/71518818/135104007-b9270067-cf6a-4d15-be18-e853160a838c.png)

   - Optimiser les 3 modèles précédemment sélectionnés.
       - recherche manuelle des hyperparamètres pour sentir leurs influences.
       - recherche automatique à partir des premières intuitions pour trouver les hyperparamètres les plus performants :
           - GridSearch CV
           - Randomized Search CV
![P4_Optimisation](https://user-images.githubusercontent.com/71518818/135103577-6f3f3c8d-3967-4605-9655-504905283a1c.png)

   - Sélectionner le modèle final le plus performant.
![P4_Model_final](https://user-images.githubusercontent.com/71518818/135104297-1f39a753-47ef-42e0-9ce6-1289af0ac4be.png)

3. **Compléments** :
    - analyser les erreurs de prédictions : sous ou sur estimation, toujours les mêmes bâtiments?
![P4_erreur_pred_elec](https://user-images.githubusercontent.com/71518818/135104379-b5f5d595-2fdc-43d6-80d6-493a2347f3bc.png)
![P4_erreur_pred_co2](https://user-images.githubusercontent.com/71518818/135104504-e23e7603-0178-4850-88b6-6e3581339a88.png)
    - voir les features importance (feature engineering efficace?)
![P4_FE](https://user-images.githubusercontent.com/71518818/135104761-05008a42-a32f-4e79-af4e-13ab192cd611.png)
    - essayer de diminuer la complexité du modèle pour obtenir de meilleurs performances (RFECV, ACP).
    - intérêt de l'Energy Star Score?
![P4_ESS](https://user-images.githubusercontent.com/71518818/135104954-7e7117ea-14dc-49a4-9ba4-9274fe348d91.png)

## Compétences
- Transformer les variables pertinentes d'un modèle d'apprentissage supervisé.
- Régression linéaire, ridge, lasso, elasticnet, SVR, decision trees, random forest, extra trees, xgboost, lightgbm, catboost, pycaret.
- Mettre en place le modèle d'apprentissage supervisé adapté au problème métier.
- Évaluer les performances d’un modèle d'apprentissage supervisé.
- Adapter les hyperparamètres d'un algorithme d'apprentissage supervisé afin de l'améliorer.

## Évaluation

![P4_eval_1](https://user-images.githubusercontent.com/71518818/135888821-f2557260-2e64-429a-8ba4-0c9ffa97deb3.png)
![P4_eval_2](https://user-images.githubusercontent.com/71518818/135888601-373c4269-1ccb-4a8c-9e35-5536f615e67c.png)

