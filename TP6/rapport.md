# RAPPORT TP6

&nbsp;

## Exercice 0 : Mise en place

&nbsp;

0.1) Voici la sortie de check_env.py :
Job 54834 sur arcadia-slurm-node-3 ; GPU : 0 ; CPU : 4
GPU 0: NVIDIA H100 NVL (UUID: GPU-c8a4c7b7-5b45-2033-9a40-92b4cf03cf33)
Machine        : arcadia-slurm-node-3
Python         : 3.10.21
torch          : 2.14.1+cu130 (CUDA 13.0)
torchvision    : 0.29.2+cu132
W1113 13:08:55.840000 24 site-packages/torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
torchao        : 0.18.0
CPU alloués    : 4  (inconnu)
Drapeaux CPU   : avx512f, avx512_vnni, avx512_bf16
gcc            : /usr/bin/gcc
GPU            : NVIDIA H100 NVL
Mémoire GPU    : 95321 Mio au total, 94795 Mio libres
bf16 supporté  : True
Données raw    : /mnt/ssd/imagenet/2012 -> (12894, 500) images (train, val)
Données small  : /tmp/rringuet/tp6/imagenet10_256 -> (12894, 500) images (train, val)
Poids ResNet-50 pré-entraînés : OK

&nbsp;

0.2) Avec juste 4 CPU pour alimenter un septième de H100, le chargement des images risque de devenir un goulot d'étranglement CPU

&nbsp;

## Exercice 1 : Mesure de référence

&nbsp;

1.1) 
Pour ref-synth:
- débit (images/s) : 719,5 images/s
- pic de mémoire : 5,95 Go
Pour ref-raw:
- débit (images/s) : 707,4 images/s
- pic de mémoire : 5,95 Go
- temps d’une époque : 20,1 s

&nbsp;

1.2)
Estimation du pic de mémoire de ref-synth :
- Paramètres fp32 : 23,5 x 10^6 x 4 octets = 94 Mo
- Gradients + états d'Adam : 23,5 x 10^6 x 12 octets = 282 Mo
- Activations : 64 images x 84 Mo/images = 5 376 Mo
Donc au total : 94 + 282 + 5 376 = 5 752 soit environ 5,75 Go. La valeur réelle est de 5,95 Go soit 200 MO de plus. Cette difféence s'explique par la mémoire utilisée par les images d'entrée et le buffers temporaires cuDNN, l'activations supplémentiares conservées pendant l'entrainement et la gestion et l'alignement des allocations mémoire par PyTorch

&nbsp;

1.3) ref-async montre le fonctionnement asynchrone de CUDA, le CPU envoie les calculs au GPU sans attndre qu'ils soient terminés ainsi avec ref-async, l'appel Python à step (x,y) prend suelement 21,5 ms alors que le calcul réel sur le GPU dure 87 ms
Un chronométraage sans torch.cuba.synchronize() est faux car sans cette fonction on mesure uniquement le temps d'envoi des tâches par le CPU et non le temps réel d'exécution sur le GPU donc le débit calculé serait incorrect

&nbsp;

1.4) Le batch maximal mesuré de 1 128 images. C'est beaucoup plus que les 120 images annoncées dans le cours. Cela s'explique pas le GPU utilisé : une NVIDIA H100 NVL, qui possède environ 94 Go de mémoire, le cours prenait plutôt comme référence un MIG de 11,5 Go. Le résultat est donc logique, avec une mémoire GPU beaucoup plus importante, on peut traiter un batch beaucoup plus grand

&nbsp;

1.5) Le débit de ref-synth est de 719,5 images/s contre 707,4 images/s pour ref-raw. La différence est très faible. Pourtant, ref-raw effectue plusieurs opérations supplémentaires comme la lecture des images JPEG, leur décodage et transformations et le transfert des données vers le GPU. Mais le débit reste quand même presque identique donc en conclusion que le chargement et le prétraitement des images ne ralentissent presque pas l'entrainement

&nbsp;

## Exercice 2 : Chargement des données et profilage

&nbsp;

2.1) 

| Tag | Données | num\_workers | pin\_memory | Débit médian (img\_s) | Min / Max (img\_s) |
| ----- | :---: | :---: | :---: | ----- | ----- |
| raw-w0 | raw | 0 | True | 231,2 img/s | 229,8 / 232,5 |
| raw-w1 | raw | 1 | True | 342,8 img/s | 339,6 / 346,0 |
| raw-w2 | raw | 2 | True | 701,2 img/s | 695,4 / 707,0 |
| raw-w3 | raw | 3 | True | 699,4 img/s | 696,8 / 701,9 |
| raw-w3-nopin | raw | 3 | False | 619,1 img/s | 615,0 / 623,3 |
| small-w0 | small | 0 | True | 262,7 img/s | 262,1 / 263,2 |
| small-w3 | small | 3 | True | 690,2 img/s | 690,0 / 690,5 |


&nbsp;

2.2) Avec 0 Worker, le débit est de 231,2 images/s car le chargement des images se fait séquentiellement. Avec 1 worker, il augmente à 342,8 images/s grâce au parallélisme. À partir de 2 workers, il atteint environ 700 images/s et ne progresse presque plus avec 3 workers. Le GPU est donc saturé et devient le facteur limitant. On ne peut pas vraiment ajouter de workers utiles, car Slurm réserve seulement 4 coeurs CPU. Le processus principal en utilise déjà 1, il en reste 3 pour les workers. En ajouter plus créerait de la concurrence entre les processus

&nbsp;

2.3) Avec pin_memory=True, le débit est de 699,4 images/s, contre 619,1 images/s avec pin_memory=False, soit une baisse sans cette option. pin_memory=True permet de place les données dans une mémoire épinglée, directement accessible par le GPU ce qui rend le transfert + rapide et permet de faire en parallèle des calculs GPU

&nbsp;

2.4) Les images de small sont déjà redimensionnées à 256 pixels, contrairement aux images oiginales de raw. Elles sont donc + rapides à charger car les fichiers sont + légers, le décodage JPEG est + rapide et les recadrages demandent moins de calculs.
Le modèle ne voit pas exactement les mêmes images. Dans raw, les images gardent leur résolution d'origine, alors que celles de small ont déjà été redimensionnées et compressées en JPEG. Les recadrages de small peuvent donc être moins nets, notamment lorsque certaines zones doivent être agrandies. Les images initialement petites ont également pu être agrandies une première fois

&nbsp;

2.5) On utilise le débit du GPU seul (ref-synth) soit 729,5 images/s pour estimer le temps de péparation des images sur le CPU. La formule utilisée est td = (1 / Dwo) - (1 / Dsynth)
 
Résultats:
- Le temps de décodage d'une image est : 2,94 ms par image pour raw et 2,42 ms par image pour small
- 1 worker peut traiter environ : 341 images/s pour raw et 414 images/s pour small
- Avec 3 wokers, le débit théorique d'alimentation atteint : 1 022 images/s pour raw et 1 241 images/S pour small

Donc si le GPUdevient + rapide, les 3 workers risquent de ne plus suffire à alimenter le GPU

&nbsp;

2.6) 

[Perfetto raw-w3](images/raw-w3.png)

[Perfetto small-w3](images/small-w3.png)

Les traces Perfetto montrent que les 2 configurations sont limitées par le calcul GPU. On observe que le GPU travaille en continu avec peu de périodes d'inactivité, le chargement des batchs par le DataLoader est rapide grâce aux 3 workers et le CPU envoie les opérations au GPU puis attend que les calculs se terminent. Les temps d'attente du CPU montrent donc que le GPU est le pricipal facteur limitant

&nbsp;

2.7) Avec la meilleure configuration (raw-w2), le débit atteint 701,2 images/s contre 719,5 images/s pourref-synth. Cela représente 701,2 / 719,5 x 100 = 97,5 %
On peut conclure que le chargement et le décodage des images sont presque entièrement masqués. Le pipeline de données arrive à alimenter le GPU sans provoquer de perte importante des performances

&nbsp;