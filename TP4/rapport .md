# RAPPORT TP4

&nbsp;

## Exercice 1: Exploration et Prétraitement des partitions ABC

&nbsp;

### Étape 1 : Chargement et découverte du jeu de données

&nbsp;

1.1.1) Le nombre exact de partitions présentes dans train.json est 214122 et dans validation.json est 2162

&nbsp;

1.1.2) Le rôle des métadonnées d'en-tête présentes sur les premières lignes avant le début des notes est de définir les paramètres globaux de la partition:
- X:1 : est le numéro de référence unique de la partition dans le fichier
- L:1/8 : est la durée par défaut d'une note sans chiffre, 1/8 est une croche
- M:4/4 : est la métrique du morceau, dans celui ci une mesure à 4 temps où chaque temps vaut une noire
- K:Emin : est la tonalité de la pièce, dans ce moraceau (Mi mineur)

&nbsp;

### Étape 2 : Construction du vocabulaire et Padding

&nbsp;

1.2.1) Il y a 95 caractères uniques. la longueur moyenne d'une partition dans le jeu d'entraînement est de 290.1 et celle  maximale est de 2968 ?

&nbsp;

1.2.2) Il indispensable d'appliquer sorted(...) sur le set des caractères uniques avant de créer char_to_index car en python, un set n'a pas d'odre fixe et change d'une execution à l'autre donc sorted(...) garantit que chaque caractère garde toujours le même identifiant numérique.
Si je sauvegardais les poids de votre modèle aujourd'hui et que vous relançais mon script demain pour générer de la musique sans ce sorted() les correspondances caractère-indice changeraient et les poids déjà appris prédiraient les mauvais caractères et génèreraient un texte incompréhensible

&nbsp;

1.2.3) On a choisi de tronquer à SEQ_LENGTH = 500 (environ deux fois la longueur moyenne) plutôt que d'aligner toutes les chansons sur la longueur maximale max_len du dataset car celà couvre largement la majorité des partitions (moyenne de 290) sans bloquer les ressources.
Un padding jusqu'à max_len causerait un temps de calcul en entrainement beaucoup plus lent par batch, des tenseurs 6 beaucoup plus volumineux risquant de saturer la VRAM et un apprentissage du modèle inefficace les partitions courtes seraient remplies de beaucoup d'espaces vides forçant le modèle à apprendre du vide plûtot que des notes

&nbsp;

## Exercice 2:  Dataset PyTorch auto-régressif et Modélisation

&nbsp;

### Étape 1 : Création du MusicDataset

&nbsp;

2.1) Si l'on fixe SEQ_LENGTH = 500 et BATCH_SIZE = 128, pour chaque partition de longueur 500, le découpage (seq[:-1] et seq[1:]) donne des séquences de 500 - 1 = 499 caractères, regroupées par lot de 128, les dimensions exactes sont:
- x.shape : torch.Size([128, 499])
- y.shape : torch.Size([128, 499])
On utilise le type torch.long (entiers 64 bits) plutôt que torch.float32 pour ces tenseurs parce que les valeurs manipulées ne sont pas des mesures continues mais des indices entiers représentant ddes catégories, en pytorch, les couches qui lisent ces identifiants pour récupérer un vecteur et la fonction de classifiaction demandent obligatoirement des entiers signés 64 bits (torch.long) pour pouvoir indexer les tableaux de poids sans ambiguïté. Des nombres à vigule (torch.float32) ne peuvent pas servir d'indices d'accès.

&nbsp;

### Étape 2 : Architecture récurrente (RNN vs LSTM)

&nbsp;

2.1) 

- Embedding: Paramètres = V x D = 95 x 256 = 24 320
- Linear (fc): Paramètres = (H x V) + V = (512 x 95) + 95 = 48 735
- RNN:
    - Poids entrée-caché : H x D = 512 x 256 = 131 072
    - Poids caché-caché : H x H = 512 x 512 = 262 144
    - Total poids: (D x H) + (H x H) = (256 x 512) + (512 x 512) = 393 216
    - Biais (b_{ih} + b_{hh}): 2 x H = 1 024
    - Total couche RNN: 393 216 + 1 024 = 394 240
- LSTM:
    - Le LSTM possède 4 portes internes (oubli, entrée, candidat, sortie) ayant chacune la même structure qu'un RNN
    - Total couche LSTM: 4 x 394 240 = 1 576 960
Le LTSM a exactement 4 fois plus de paramètres que le bloc RNN car il calcule 4 transformations linéaires indépendantes au lieu d'une seule 

&nbsp;

2.2) Il bien plus pertinent d'utiliser une couche nn.Embedding en entrée plutôt que de passer directement l'indice scalaire i dans {0, ..., V-1\} dans une couche linéaire 1 -> embedding_dim} pour éviter un ordre faux, passe une simple ciffre forcerait le modèle à croire que le caractère numéro 10 vaut 2 fois le caratère 5 et l'embedding associe à chaque symbole son propre vecteur de 256 nombres, lui permettant d'apprendre des nuances sans contraintes mathématique

&nbsp;

2.3) Pour nn.RNN, hidden_state est un tenseur unique de forme (num_layers, batch_size, hidden_size). Pour nn.LSTM, hidden_state est un tuple de deux tenseurs : cell state qui est la mémoire à long terme et hidden state qui est la mémoire à court teme et la sortie directe à l'instant t

