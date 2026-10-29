# RAPPORT TP5

&nbsp;

## Exercice 1: Chargement des données et Embeddings

&nbsp;

1.1) Redimensionner explicitement toutes les images à une taille fixe de (224, 224) avant de les passer dans le réseau, plutôt que de garder leur résolution d'origine est nécessaire pour plusieurs raisons:
- le réseau ResNet est conçu pour traiter des images de cette taille, qui correspond à son format d'entrée habituel
- les redimensionner permet d'avoir des images de dimensions identiques
- Cela permet de standadiser les données et de simplifier

&nbsp;

## Exercice 2: Extraction de caractéristiques et Transfer Learning

&nbsp;

2.1) Il est crucial de geler (requires_grad = False) les poids du ResNet dans ce TP pour 2 raisons:
- matérielle : ça réduit la consommation de mémoire GPU et accélère l'entrainement car les gradients du ResNet ne sont pas calculés.
- algorithmique: Pour préserver les filtres visuels déjà appris lors du pré-entrainement sur ImageNet, en évitant que les mises à jour des poids ne dégradent ces caractéristiques

&nbsp;

2.2) Si on passe un tenseur image de forme (32, 3, 224, 224) (batch_size, channels, H, W) dans cnn_features, la dimension (shape) exacte du tenseur de sortie sera (32, 2048, 7, 7). 32 c'est le nombre d'images du batch, 2048 c'est les cartes caractéristiques, et les dimensions 7 x 7 à leur taille spatiale

&nbsp;

