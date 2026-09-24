# RAPPORT TP2

&nbsp;

## Exercice 1: Création d'un dataset personnalisé

1.1) Utiliser StandardScaler sur l'ensemble du dataset, avant le split est une mauvaise pratique en apprentissage automatique car c'est une forme de data leakage. En effet, StandardScaler calcule la moyenne et l'écart-type sur tout le dataset donc à partir des données du train, de la validation et du test. Ainsi le modèle bénéficie d'informations (données de val et de test) qui devraient rester inconnues jusqu'à l'évaluation et ces dernières de plus ont un impact indirect sur la transformation appliquée aux données d'entrainement

&nbsp;

1.2) Si le dataset était trop volumineux pour tenir dans la RAM, J'utiliserai la classe IterableDataset à la place de Dataset

## Exercice 2: MLP et Régularisation L1 / L2

&nbsp;

2.1)

&nbsp;

2.2)

&nbsp;

2.3)

&nbsp;

