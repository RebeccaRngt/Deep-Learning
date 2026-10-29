# RAPPORT TP5

&nbsp;

## Exercice 1: Chargement des données et Embeddings

&nbsp;

1.1) Redimensionner explicitement toutes les images à une taille fixe de (224, 224) avant de les passer dans le réseau, plutôt que de garder leur résolution d'origine est nécessaire pour plusieurs raisons:
- le réseau ResNet est conçu pour traiter des images de cette taille, qui correspond à son format d'entrée habituel
- les redimensionner permet d'avoir des images de dimensions identiques
- Cela permet de standadiser les données et de simplifier

&nbsp;