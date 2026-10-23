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

&nbsp;

## Exercice 3: Entraînement, Dynamique des Gradients et Comparaison RNN vs LSTM

&nbsp;

3.1) On doit écrire logits.reshape(-1, vocab_size) et y.reshape(-1) avant d'appeler loss_fn car la fonction nn.crossEntropy de Pytorch demande un tenseur de prédictions à 2 dimensions et un tenseur cible à une dimension, comme nos pédictions sortent du modèle sous 3 dimensions (batch_size, seq_len, vocab_size) et nos cibles à 2 dimensions (batch_size, seq_len), on regroupe le lot et la longueur de séquence en une unique dimension de N échantillons
La valeur numérique de la dimension obtenue à la place du -1 est 128 x 499 = 63 872, et représente le nombre total de prédictions de caractères individuels que le modèle doit évaluer simultanément sur l'ensemble du lot

&nbsp;

3.2) La fonction clip_grad_norm_ doit impérativement être appelée après loss.backward() mais avant optimizer.step() pour respecter l'ordre de mise à jour des paramètres, elle intervient après loss.backward() parce que les gradients n'esistent tout simplement pas avant la rétropropagation : c'est elle qui les calcule et les place dans l'attribut .grad de chaque poids et elle vient avant optimizer.step() car l'optimiseur utilise ces gradients pour modifier directement les poids du modèle, il faut donc borner leur norme avant cette mise à jour pour pas que les paramètres ne divergent

&nbsp;

3.3) 

[Courbes comparatives TensorBoard Loss/Validation](images/loss_val.png)

&nbsp;

[Courbes comparatives TensorBoard Accuracy/Validation](images/acc_val.png)

&nbsp;

[Courbes comparatives TensorBoard GradNorm/step](images/gradnorm_step.png)

&nbsp;

Sur le graphique GradNorm/step, on remarque que la norme des gradients dépasses 1,5 au début de l'entrainement pour les 2 RNN. Sans gradient clipping, la perte de validation varie beaucoup surtout jusqu'à l'époque 7, avec une seuil de clipping de 1,0, la courbe devient plus stable. Le problème théorique du RNN simple que le gradient clipping corrige est l'explosion du gradient ce qui arrive souvent dans les RNN quand ils traitent de longeus séquences, ici de 500 caractères
Le LSTM apprend plus rapidemment que le RNN, dès la première époque , sa perte de validation est inférieure à 0,80. A l'époque 15, le LSTM obitent de meilleurs résultats:
- Loss : 0,56 contre 0,74 pour le RNN
- Accuracy : 82% contre 77% pour le RNN
Le LSTM parvient à capturer des dépendances plus longues dans la partition musicale. Il y a le problème de disparition du gradient dans les RNN simples le LSTM utilise l'équation de mise à jour du cell state. Ce mécanisme permet de mieux conserver les informations sur de longues séquences et donc de mieux apprendre les structures musicales

&nbsp;

3.4) L'Accuracy/Validation affichée est artificiellement optimiste quant à la capacité réelle du modèle à prédire des notes de musique car la partition contient environ 250 caractères, mais les séquences ont une longeur fixe de 500 caractères. Le reste est donc rempli avec des espaces (' '). Comme ces espaces sont faiciles à prédire, le modèle obtient beaucoup de bonnes réponses sans forcément bien prédire les notes. L'accuracy affichée est donc lus élevée que la performance réell sur les caractères musicaux
La modification simple de nn.CrossEntropyLoss qui permettrait de ne pas compter le padding dans la perte  est:
loss_fn = nn.CrossEntropy(ignore_index=char_to_index[" "])
Cette modification permet d'ignorer les espaces lors du calcul de la perte

&nbsp;

## Exercice 4:  Génération musicale, Température et Job Slurm

&nbsp;

### Étape 1 : Implémentation de la génération avec propagation de l'état caché

&nbsp;

4.1.1) Sans conserver le hiden_state, le modèle doit recalculer toute la séquence à chaque nouveau caractère généré et ça demande beaucoup de calculs, avec un complexité quadratique. Avec notre methode, on conserve le hidden_state et ke modèle traite seulement un nouveu caractère à chaque étape. Cela permet de diminuer les calculs et d'obtenir une compléxité linéaire pour la génération, elle est plus rapide pour générer des longues séquences

&nbsp;

4.1.2) La température T permet de contrôler le choix des caractères générés par le modèle
- Quand (T \to 0^+): le modèle choisit presque toujours le caractère ayant la probabilité la plus élevée, le résultat est donc plus prévisible et moins varié
- Quand (T T \to +\infty): les probabilités des caractères deviennent presque égales, le modèle choisit alors les caractères de manière presque aléatoire, ce qui donne des résultats plus variés, mais souvent moins cohérents.
Donc une température faible favorise la cohérence et une élevée augmente la diversité des caractères générés

&nbsp;

4.1.3) 

- 1ère partition:

 Température T = 0.2
X:1
T:Irish Reel
M:4/4
L:1/8
K:D
 A | F2 AF E2 FE | D2 FD E2 E2 | F2 AF E2 E2 | F2 A2 A2 F2 | E2 E2 E2 F2 | D2 D2 D2 D2 | 
 D2 D2 D2 F2 | E2 E2 E2 F2 | D2 D2 D2 D2 | D2 D2 D2 D2 | E2 E2 E2 A2 | F2 A2 A2 F2 | E2 E2 E2 F2 | 
 A2 F2 E2 D2 | D2 D2 D2 :|

La partition respecte globalement la synthaxe ABC avec des notes valides et des barres de mesure mais certaines mesures ne respectent pas exactement le rythme 4/4. Musicalement, je remarque beaucoup de répétitions, (des notes D2 et E2 notamment) après la mélodie reste assez cohérente, mais elle manque de variété

- 2e partition:

 Température T = 0.8
X:1
T:Irish Reel
M:4/4
L:1/8
K:D
 AG | A2 AG ED EF | G2 ED CD ED | A,C E2 A,D D2 :: BBBA BcdB | AF F2 dFFD | B,EE EDB, B,2 :|

Cette partition a une meilleure diversité musicale. Les notes sont plus variées et la mélodie est moins répétitives (c'est celle que je préfère). La syntaxe ABC est globalement correcte, même si certaines mesures sont incomplètes ou contiennent trop de notes, le résultat est plus naturel et plus intéressant

- 3e partiton:

 Température T = 2.0
X:1
T:Irish Reel
M:4/4
L:1/8
K:D
 eege{a} f!p!a2N<decTwBGh e2TGF{e}dAGFga/g}9Ag z"Ab0<g[ $urnacudL]" exTcA D2fa | !0ulfrmate{6evdeYe."Fsbacfg [f9]Mdnc BoAG F.d.d.A !ferm]edagg!TagfMf[M:^165] */a3/2Wa2g- f6-TE82] | 
 cDCA{Lfe}[fDdGdf.E8!ufuHg5eO (vvA-C]B o-),E) !we[aarde!M .A2 | e(Hc-B)A"^*)"{F} FADF{AAc}d4>c |{B2:O3/8 mdA(8a3)z/4g 

Avec cette température, la partition devient désordonnée, on retrouve des caractères et des symboles qui ne respectent pas la syntaxe ABC. Les mesures ne sont plus correctement structurées, une partie de la partition n'est pas lu par ABC Player car jugée indéfinie. Musicalement, c'est chaotique et ça ne ressemble plus à une mélodie cohérente

Donc la température influence directement la qualité des partitions générées. La témpérature T = 0.8 est le meilleur compromis entre variété des notes et cohérence de la musique


&nbsp;

### Étape 2 : Entraînement à plus grande échelle via sbatch

&nbsp;

4.c) Pour augmenter les données, on peut transposer les partitions dans différentes tonalités en décalant les notes de quelques demi-tons. Pour ça, il faut modifier la tonalité dans l'en-tête K: et adapter les notes de partition, prendre en compte les octaves, les altérations en gardant les durées et barres de mesures
Cette methode permet de créer plsu de données d'entrainement à partir des partitions existantes, sans changer leur structure musicale, le modèle apprend donc les mêmes mélodies dans plusieurs tonalités, au lieu de mémoriser uniquement des suites de notes précises. Celà permet de limiter le sur-apprentissage et d'aider le modèle à mieux générer de nouvelles partitions