# RAPPORT TP3

&nbsp;

## Exercice 1: Préparation d'ImageNet et Augmentation de données

1.1) La normalisation met les valeurs des pixels sur une échelle plus adaptée à l'apprentissage. Donc si on oublie l'étape transforms.Normalize, les valeurs des pixels peuvent avoir une distribution différente de celle attendue, ce qui peut rendre la descente de gradient moins stable ou ralentir la convergence du modèle.
Les moyennes et les écarts-types doivent être calculés uniquement sur le jeu d'entrainement pour éviter tout fuite d'informations du jeu de validation vers l'apprentissage. On utilise ensuite ces mêmes statistiques pour normaliser les deux jeux.

&nbsp;

1.2) On utilise RandomResized et RandomHorizontalFlip lors de l'entrainement pour créer des variations aléatoires des images afin que le modèle soit plus robuste, mieux généraliser et afin d'éviter le surapprentissage mais Resize + CenterCrop des transformations déterministes (sans flip) sur le jeu de validation pour que les images soient traitées de la même manière à chaque évaluation pour permettre de comparer les performances du modèle de façon fiable 

&nbsp;

1.3) L'augmentation par RandomHorizontalFlip serait néfaste et fausseait les étiquettes dans le cadre où cette augmentation ne garde pas le même label par exemple pour la classification des panneaux de signalisation, un RandomHorizontalFlip peut être néfaste, un panneau d'interdiction de tourner à droite devient une panneau d'interdiction de tourner à gauche l'image est associée à une mauvaise étiquette ce qui peut perturber l'apprentissage du modèle