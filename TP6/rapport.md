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

