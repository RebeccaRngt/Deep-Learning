# RAPPORT TP3

&nbsp;

## Exercice 1: Préparation d'ImageNet et Augmentation de données

1.1) La normalisation met les valeurs des pixels sur une échelle plus adaptée à l'apprentissage. Donc si on oublie l'étape transforms.Normalize, les valeurs des pixels peuvent avoir une distribution différente de celle attendue, ce qui peut rendre la descente de gradient moins stable ou ralentir la convergence du modèle.
Les moyennes et les écarts-types doivent être calculés uniquement sur le jeu d'entrainement pour éviter tout fuite d'informations du jeu de validation vers l'apprentissage. On utilise ensuite ces mêmes statistiques pour normaliser les deux jeux.

&nbsp;

1.2) On utilise RandomResized et RandomHorizontalFlip lors de l'entrainement pour créer des variations aléatoires des images afin que le modèle soit plus robuste, mieux généraliser et afin d'éviter le surapprentissage mais Resize + CenterCrop des transformations déterministes (sans flip) sur le jeu de validation pour que les images soient traitées de la même manière à chaque évaluation pour permettre de comparer les performances du modèle de façon fiable 

&nbsp;

1.3) L'augmentation par RandomHorizontalFlip serait néfaste et fausseait les étiquettes dans le cadre où cette augmentation ne garde pas le même label par exemple pour la classification des panneaux de signalisation, un RandomHorizontalFlip peut être néfaste, un panneau d'interdiction de tourner à droite devient une panneau d'interdiction de tourner à gauche l'image est associée à une mauvaise étiquette ce qui peut perturber l'apprentissage du modèle

&nbsp;

## Exercice 2: Étude théorique et implémentation des architectures

### Étape 1 : Implémentation du MLP

&nbsp;

2.a.1)

##### - Sans biais

En entrée une image à 3 canaux (RGB), une hauteur et une largeur de 227 pixels soit nombre de neurones en entrée, input_size = 3 x 227 x 227 = 154 587 
Pour la 1ère couche: Nombre de poids couche 1 (sans biais) = 154 587 x 256 = 39 574 272
Pour la 2e couche: nombre de neurones en entrée et en sortie, hidden_size = 256 (car le MLP a 2 couches cachées de 256 neurones)  donc ainsi nombre de poids couche 2 (sans biais) = 256 x 256 = 65 536 poids
Pour la 3e couche: nombre de neurones en entrée, hidden_size = 256 et nombre de neurones en sortie, num_classes = 1 000 donc ainsi nombre de poids couche 3 (sans biais) = 256 x 1 000 = 256 000
On additionne le nombre de poids de chaque couche pour avoir le nombre de poids total ainsi nombre de poids total (sans biais) = 39 574 272 + 65 536 + 256 000 = 39 895 808 poids
&nbsp;

##### - Avec biais

Pour la 1ère couche: Il y a un biais par neurone donc 256 biais ainsi nombre de poids couche 1 (avec biais) = nombre de poids couche 1 (sans biais) + 256 = 39 574 272 + 256 = 39 574 272 poids
Pour la 2e couche: Elle donne 256 neurones donc 256 biais ainsi nombre de poids couche 2 (avec biais) = nombre de poids couche 2 (sans biais) + 256 = 65 536 + 256 = 65 792 poids
Pour la 3e couche: Elle donne 1 000 neurones donc 1 000 biais ainsi nombre de poids couche 3 (avec biais) = nombre de poids couche 3 (sans biais) + 1000 = 256 000 + 1 000 = 257 000 poids
0n additionne le nombre de poids de chaque couche pour avoir le nombre de poids total ainsi nombre de poids total (avec biais) = 39 574 528 + 65 792 + 257 000 = 39 897 320 poids

La couche qui concentre plus de 99% des paramètres est la 1ère, car elle doit connecter chaque pixel de l'image aux 256 neurones cachés

&nbsp;

2.a.2) Le défaut structurel majeur du MLP lorsqu'on lui fournit une image aplatie (flatten) est qu'il écrase la structure spatiale de l'image en entrée et l'aspect spatial est très important, il ne tient pas naturellement compte de la proximité des pixels ni de la position des objets

&nbsp;

### Étape 2 : Analyse et implémentation d'AlexNet

&nbsp;

2.b.1)

| Couche | Taille entrée (C, H×W) |  | Hyperparamètres (Filtres/Neurones, K, S, P) |  |  |  | Taille sortie (C, H×W) |  | \# Paramètres (avec biais) | Mémoire sortie (KB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv1** \+ ReLU | 3 | 227×227 | 64 | 11×11 | S=4 | P=2 | 64 | 56×56 | 23 296 | 784 |
| **MaxPool1** | 64 | 56×56 | \- | 3×3 | S=2 | P=0 | 64 | 27×27 | 0 | 182,25 |
| **Conv2** \+ ReLU | 64 | 27×27 | 192 | 5×5 | S=1 | P=2 | 192 | 27×27 | 307 392 | 546,75 |
| **MaxPool2** | 192 | 27×27 | \- | 3×3 | S=2 | P=0 | 192 | 13×13 | 0 | 126,75 |
| **Conv3** \+ ReLU | 192 | 13×13 | 384 | 3×3 | S=1 | P=1 | 384 | 13×13 | 663 936 | 253,5 |
| **Conv4** \+ ReLU | 384 | 13×13 | 256 | 3×3 | S=1 | P=1 | 256 | 13×13 | 884 992 | 169 |
| **Conv5** \+ ReLU | 256 | 13×13 | 256 | 3×3 | S=1 | P=1 | 256 | 13×13 | 590 080 | 169 |
| **MaxPool3** | 256 | 13×13 | \- | 3×3 | S=2 | P=0 | 256 | 6×6 | 0 | 36 |
| **Flatten** \+ Dropout | 256 | 6×6 | p \= 0.5 |  |  |  | 9216 (1D) |  | 0 | 36 |
| **Linear1** \+ ReLU \+ Dropout | 9216 |  | 4096 neurones (p \= 0.5) |  |  |  | 4096 |  | 37 752 832 | 16 |
| **Linear2** \+ ReLU | 4096 |  | 4096 neurones |  |  |  | 4096 |  | 16 781 312 | 16 |
| **Linear3** (Sortie) | 4096 |  | 1000 classes |  |  |  | 1000 |  | 4 097 000 | 3,91 |

&nbsp;

2.b.2) Dans AlexNet, l'essentiel des paramètres (poids) se concentre pricipalement dans les couches linéaires, en particulier Linear1. A l'inverse, l'essentiel de la mémoire des activations se concentre principalement dans les premières couches convolutionnelles, notamment Conv1 et Conv2

&nbsp;

2.b.3) Le MLP possède envirion 39 millions de paramètres et utilise environ 610 KB d'activations par image, contre 61 millions de paramètres et 2 440 KB d'activations pour AlexNet. AlexNet utilise plus de mémoire car ses couches convultionnelles produisent de grandes cartes de caractéristiques contenant beaucoup de valeurs interédiaires contrairement au MLP qui a des couches cachées de 256 neurones seulement. Donc le nombre de paramètres dépend des connexions et des poids du réseau, tandis que la mémoire des activations dépend surtout de la taille des sorties intermédiaires à conserver pendant l'apprentissage

&nbsp;