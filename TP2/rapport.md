# RAPPORT TP2

&nbsp;

## Exercice 1: Création d'un dataset personnalisé

1.1) Utiliser StandardScaler sur l'ensemble du dataset, avant le split est une mauvaise pratique en apprentissage automatique car c'est une forme de data leakage. En effet, StandardScaler calcule la moyenne et l'écart-type sur tout le dataset donc à partir des données du train, de la validation et du test. Ainsi le modèle bénéficie d'informations (données de val et de test) qui devraient rester inconnues jusqu'à l'évaluation et ces dernières de plus ont un impact indirect sur la transformation appliquée aux données d'entrainement

&nbsp;

1.2) Si le dataset était trop volumineux pour tenir dans la RAM, J'utiliserai la classe IterableDataset à la place de Dataset

&nbsp;

## Exercice 2: MLP et Régularisation L1 / L2

&nbsp;

2.1) Avec l1_lambda = 0.1 et l2_lambda = 0, Loss devient Loss = base_loss + 0.1 * l1_penality. La pénalité L1 pend plus d'importance dans la Loss et pousse les poids du réseau vers 0. Le modèle va préférer régulariser fortement en réduisant ses poids plutôt que d'apprendre des rélations complexes sur les données ce qui limite sa capacité à apprendre correctement, ce phénomène d
s'appelle sous-apprentissage (underfitting)

&nbsp;

2.2) L'argument de l'optimisateur (ex: optim.SGD) qui permet d'appliquer cette régularisation L2 automatique est weight_decay

&nbsp;

2.3) La différence conceptuelle sur les poids du réseau entre la régularisation L1 et L2 vient de la forme de la pénalité. La régularisation L1 favorise la sparsité, elle pousse certains poids exactement vers 0 tandis que la régularisation L2 tend à rétrécir les poids sans nécéssairement les mettre à 0

&nbsp;

## Exercice 3: Comparaison des Optimiseurs et TensorBoard

&nbsp;

3.1) [Capture d'écran de Tensorboard des courbes de perte des 4 optimiseurs](images/courbes_optimiseurs.png)

&nbsp;

3.2) L'optimiseur qui converge le plus rapidement initialement est RMSprop.

&nbsp;

3.3) La courbe de l'optimiseur SGD descend très lentement pour stagner vers 0,64 à la fin tandis que pour celle de l'optimiseur Momentum la pente est bien plus prononcé dès les premières epochs et atteint une perte finale d'environ 0,58. L'effet de l'ajout du Momentum sur la descente de gradient est le suivant: Le Momentum garde en mémoire la direction des gradients précédents, ce qui crée une sorte d'inertie et permet donc d'accélérer dans la bonne direction, de réduire les zigzags et de rendre la descente vers le minimum plus rapide et plus stable

&nbsp;

## Exercice 4: Analyse des métriques (Précision, Rappel, F1, AUC)

&nbsp;

4.1)
Précision (Precision) : représente la part de prédictions positives qui sont correctes

$Precision\ =\ \frac{TP}{TP\ +\ FP}$

Rappel (Recall) : mesure la proportion d'éléments réellement positifs qui ont été correctement identifiés par le modèle

$Recall\ =\ \frac{TP}{TP\ +\ FN}$

&nbsp;

4.2) Dans le contexte médical (détecter une maladie cardiovasculaire), il vaut mieux privilégier un fort rappel car on veut éviter de ne pas décter une personne réellement malade (faux négatif), qu'un faux positif peut entrainer des examens supplémentaires

&nbsp;

4.3) L'aire sous la courbe ROC (AUCpar rapport aux autres métriques calculées à un seuil fixe de 0,5) mesure la capacité du modèle à distinger les malades des non-malades pour différents seuils pas seulement 0,5. Plus elle est grande, meilleur est le modèle