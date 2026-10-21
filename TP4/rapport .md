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