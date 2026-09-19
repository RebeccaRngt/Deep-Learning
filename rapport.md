# RAPPORT TP1

1.c) Le modèle exact du GPU qui m’a été alloué est NVIDIA L4 avec 24 Go de VRAM

&nbsp;

1.d) La commande exacte que j’ai tapée pour annuler mon job est : scancel 1547

&nbsp;

1.e) Le nom exact du fichier de log généré dans le dossier **logs/** est hello-slurm-1551.out (il y a également un fichier .err)

&nbsp;

1.f) ReqMem est la RAM demandée tandis que la MaxRSS correspond au pic maximal de RAM réellement utilisé à un instant T

&nbsp;

2.b)

- La commande pour vérifier la version exacte de Python installée dans notre environnement actif est python \--version et le résultat renvoyé est: Python 3.10.21  
- La commande pour vérifier le chemin du binaire utilisé est which python et le résultat renvoyé est: /mnt/hdd/homes/rringuet/miniforge3/envs/deeplearning/bin/python

&nbsp;

2.d) La sortie du script check\_gpu.py est la suivante:

PyTorch version: 2.13.0

CUDA available: False

Attention, aucun GPU détecté \!

CUDA available retourne False et 2 raisons pourraient possiblement expliquer le problème:

- le job a été lancé sans réserver de GPU (pas de par exemple \--gres \=gpu:1) avec SLURM  
- Le driver NVIDIA n’est pas installé

&nbsp;

2.f) La commande qui permet d'afficher la version de TensorBoard que l’on vient d’installer est: tensorboard \--version

&nbsp;

3.a)

! [Dessin du MLP][images/mlp.png]

Sans biais

1ère couche: 3 neurones d’entrée connectés à 4 neurones cachés donc 12 poids

* $3\times 4\ =\ 12\ poids$

2e couche: 4 neurones de la couche cachée connectés à 2 neurones de sortie donc 8 poids

* $4\times 2\ =\ 8\ \ poids$

Nbr de paramètres \= Nbr de poids de la 1ère couche \+ Nbr de poids de la 2e couche

* $Nombre\ de\ paramètres\ total\ =12\ +\ 8=\ 20\ \ paramètres$

&nbsp;

Avec biais

on rajoute les 4 biais pour les neurones cachés et les 2 biais pour les neurones de sorties, ainsi avec les biais le nombre de paramètres change:

&nbsp;

* $Nombre\ de\ paramètres\ total\ =20\ +\ 4\ +\ 2=\ 26\ \ paramètres$

&nbsp;

3.b)

$Z\ =\ X\cdot {W}_{1}^{T}+{b}_{1}$

$H\ =\ ReLU\ (X\cdot {W}_{1}^{T}+{b}_{1})\ =\ ReLU(Z)$

$Y\ =H\cdot {W}_{2}^{T}+{b}_{2}=\ ReLU(Z)\cdot {W}_{2}^{T}+{b}_{2}=ReLU\ (X\cdot {W}_{1}^{T}+{b}_{1})\ \cdot {W}_{2}^{T}+{b}_{2}$

&nbsp;

Dimensions :

X  : (N, 3\)

W1 : (4, 3\)

b1 : (1, 4\) \-\> diffusé en (N, 4\)

H  : (N, 4\)

W2 : (2, 4\)

b2 : (1, 2\) \-\> diffusé en (N, 2\)

Y  : (N, 2\)

&nbsp;

3.c)

![Reponse question 3.c partie 1][images/q3c1.png]

![Reponse question 3.c partie 2][images/q3c2.png]

&nbsp;

3.d)

![Reponse question 3.d][images/q3d.png]

3.e) On utilise la règle de la chaîne car un réseau de neurones profond est composé de plusieurs couches et chacune réalise une fonction qui dépend de la précédente. Pour calculer l’impact d’un poids sur l’erreur finale, il faut multiplier les dérivées obtenues à tavers les différentes couches. La règle de la chaîne permet donc de calculer les gradients de chaque poids ce qui permet de mettre à jour les paramètres du réseau avec la descente de gradient.

On utilise les mini-batchs car ils permettent de trouver un bon équilibre entre le calcul sur un seul exemple et sur l’ensemble des données. Utiliser un seul exemple donne un gradient très burité, tandis qu’utiliser toutes les données à chaque étape demande beaucoup de temps et de mémoire. Avec un mini-batch, on calcule le gradient sur un petit groupe d’exemples, ce qui rend l’apprentissage plus rapide et plus stable, tout en permettant d’utiliser efficacement les GPU (opérations matricielles en parallèle) et de limiter la mémoire necessaire.

&nbsp;

3.f)

&nbsp;

| Tâche | Fonction finale (Sortie) | Fonction de perte (Loss) |
| :---- | :---- | :---- |
| Classification binaire | 1\. Sigmoïde | A. Binary Cross-Entropy |
| Classification multi | 2\. Softmax | B. Cross-Entropy |
| Régression pure | 3\. Identité (aucune) | C. MSE (Mean Squared Error) |

&nbsp;

&nbsp;

4.a) L’argument batch\_size correspond au nombre d’exemples que le modèle traite avant de mettre à jour ses paramètres. L’argument shuffle permet de mélanger les données à chaque époque. Il est mis à True pour l’entrainement pour que le modèle ne voie pas toujours les exemples dans le même ordre pour un apprentissage plus efficace. Pour le test, on met shuffle à False car les données n’ont pas besoin d’être mélangées, on cherche juste à évaluer le modèle.

&nbsp;

4.b) torch.flatten(x, 1\) permet de transformer chaque image en un vecteur de valeurs pour pouvoir l’envoyer à la couche linéaire (nn.Linear) car celle-ci attend en entrée des données sous forme de vecteurs. Le 1 permet de garder la dimension du batch en regroupant les autres dimensions. Donc dans un batch de 32 images chaque image passe de (3, 32, 32\) à un vecteur de 3072 valeurs (3 x 32 x 32 \= 3072\) et donc le batch passe de (32, 3, 32, 32\) à une matrice de taille (32, 3072\) où chaque ligne représente une image sous forme d’un vecteur.

Il ne faut pas ajouter Softmax avant nn.CrossEntropyLoss car cette fonction de perte applique déjà automatiquement le Softmax.

&nbsp;

4.d) On utilise torch.no\_grad() los de l’évaluation car on ne cherche plus à entrainer le modèle juste à faire des prédictions. On a donc plus besoin de calculer et garder les gradients pour la rétropopagation. On réduit l’utilisation de la mémoire (GPU) et les calculs d’évaluation sont plus rapides.

CIFAR-10 contient 10 classes donc si le classificateur prédit les classes de manière aléatoire, la probabilité de trouver la bonne classe pour chaque image est de 1/10, on s’attendrait donc à une accuracy d’environ 10% sur le jeu de test

&nbsp;

5.a) Il est important d’inclure la date, l’heure et les hyperparamètres dans le nom du dossier de logs (run\_name) pour pouvoir identifier facilement chaque expérience, quand elle a été réalisé et avec quels paramètres (taille du batch bs et learning rate lr). Mais aussi retrouver les résultats d’une expérience précise, éviter de mélanger ou d’écraser les logs

&nbsp;

5.d)  Au niveau 0,98 de smoothing, on peut clairement distinguer la tendance  de la courbe Loss/train\_step sans masquer des changements importants. On observe plus de bruit sur Loss/train\_step comparativement à Loss/train car  Loss/train\_step correspond à la perte calculée sur chaque mini-batch et comme chaque mini-batch contient une petite partie des données, la perte peut varie fortement d’un batch à l’autre. Loss/train repésente une moyenne de la perte sur l’ensemble de l’époque, donc moins de variations ainsi la courbe est plus stable et lisse.

&nbsp;

5.e) On observe que le run (bs128, lr=0.1) se comporte différemment des deux autres. Sa Loss/train diminue régulièrement passant d’environ 1,7 à 1,3 tandis que sa Loss/val reste beaucoup plus faible, autour de 1,6 ce qui montre que le modèle apprend correctement et obtient de meilleurs performances sur les données de validation. Pour les runs (bs32, lr \= 0.001) et (bs32, lr=0.01), la Loss/train reste autour de 2,1 et la Loss/val est également plus élevée autour de 2,2. Leu apprentissage est donc moins performant. Pour la courbe Accuracy/val, le run (bs128, lr=0.1) atteint la meilleure accuracy de validation avec environ 48% à la dernière époque contre 36-37% environ pour les deux autres

On detecte visuellement un sur-apprentissage lorsque la Loss/train continue de diminuer alors que la Loss/val se met à augmenter après avoir été en diminution. Le modèle commence à apprendre par coeur les données d’entrainement et comment de moins en moins d’erreur lors du train mais n’arrive plus à s’adapter et à prédire correctement sur de nouvelles données d’où le fait que pour l’évaluation les erreurs augmentent.
