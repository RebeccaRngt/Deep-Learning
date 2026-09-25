# RAPPORT TP2

&nbsp;

## Exercice 1: Création d'un dataset personnalisé

1.1) Utiliser StandardScaler sur l'ensemble du dataset, avant le split est une mauvaise pratique en apprentissage automatique car c'est une forme de data leakage. En effet, StandardScaler calcule la moyenne et l'écart-type sur tout le dataset donc à partir des données du train, de la validation et du test. Ainsi le modèle bénéficie d'informations (données de val et de test) qui devraient rester inconnues jusqu'à l'évaluation et ces dernières de plus ont un impact indirect sur la transformation appliquée aux données d'entrainement

&nbsp;

1.2) Si le dataset était trop volumineux pour tenir dans la RAM, J'utiliserai la classe IterableDataset à la place de Dataset

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

3.1)  ! [Capture d'écran de Tensorboard des courbes de perte des 4 optimiseurs][images/courbes_optimiseurs.png]

&nbsp;

3.2) L'optimiseur qui converge le plus rapidement initialement est RMSprop.

&nbsp;

3.3) La courbe de l'optimiseur SGD descend très lentement pour stagner vers 0,64 à la fin tandis que pour celle de l'optimiseur Momentum la pente est bien plus prononcé dès les premières epochs et atteint une perte finale d'environ 0,58. L'effet de l'ajout du Momentum sur la descente de gradient est le suivant: Le Momentum garde en mémoire la direction des gradients précédents, ce qui crée une sorte d'inertie et permet donc d'accélérer dans la bonne direction, de réduire les zigzags et de rendre la descente vers le minimum plus rapide et plus stable