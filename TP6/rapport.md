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

## Exercice 3 : Précision mixte, channels_last et torch.compile

&nbsp;

3.1) 
| Configuration | Données | Débit médian (img/s) | Min (img/s) | Max (img/s) | Pic mémoire (Go) | Compilation (s)|
| :---: | ----- | ----- | ----- | ----- | ----- | ----- |
| fp32-synthetic | donnees=synthetic | img_s=717.9 | img_s_min=716.7 | img_s_max=718.4 | pic_memoire_go=5.95 | compilation_s=nan |
| tf32-synthetic | donnees=synthetic | img_s=1646.1 | img_s_min=1629.5 | img_s_max=1672.7 | pic_memoire_go=5.93 | compilation_s=nan |
| bf16-synthetic | donnees=synthetic | img_s=2008.1 | img_s_min=2007.8 | img_s_max=2008.1 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-synthetic | donnees=synthetic | img_s=3349.8 | img_s_min=3246.4 | img_s_max=3413.9 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-compile-synthetic | donnees=synthetic | img_s=3602.9 | img_s_min=3570.6 | img_s_max=3940.3 | pic_memoire_go=3.07 | compilation_s=49.6 |
| bf16-cl-raw-w1 | donnees=raw | img_s=282.6 | img_s_min=261.1 | img_s_max=298.1 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-compile-raw-w1 | donnees=raw | img_s=226.5 | img_s_min=205.0 | img_s_max=232.0 | pic_memoire_go=3.07 | compilation_s=5.8 |
| bf16-cl-raw-w3 | donnees=raw | img_s=1404.5 | img_s_min=715.7 | img_s_max=2505.3 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-compile-raw-w3 | donnees=raw | img_s=3515.7 | img_s_min=3436.7 | img_s_max=3524.1 | pic_memoire_go=3.07 | compilation_s=5.6 |
| bf16-cl-small-w1 | donnees=small | img_s=498.6 | img_s_min=487.3 | img_s_max=500.4 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-compile-small-w1 | donnees=small | img_s=330.0 | img_s_min=326.5 | img_s_max=381.8 | pic_memoire_go=3.07 | compilation_s=4.2 |
| bf16-cl-small-w3 | donnees=small | img_s=2673.2 | img_s_min=2623.3 | img_s_max=2686.6 | pic_memoire_go=3.29 | compilation_s=nan |
| bf16-cl-compile-small-w3 | donnees=small | img_s=1387.7 | img_s_min=1106.5 | img_s_max=2809.2 | pic_memoire_go=3.07 | compilation_s=5.5 |


&nbsp;

3.2) Le débit est de 719,9 images/s pour fp32-synthetuc contre 719,5 images/s pour ref-synth, soit un écart très très faible. Les mesures sont donc très stables, même sur 2 noeuds différents

&nbsp;

3.3) 

| Optimisation | Débit (img/s) | Gain par rapport à l'étape précédente | Gain cumulé |
| ----- | :---: | :---: | :---: |
| fp32 | 717,9 | - | - |
| + TF32 | 1 646,1 | x 2,29 | x 2,29 |
| + bf16 | 2 008,1 | x 1,22 | x 2,80 |
| + channels_last | 3 349,8 | x 1,67 | x 4,67 |
| + torch.compile | 3 602,9 | x 1,08 | x 5,02 |

Toor, corps
- Tensor Cores (tf32): Les calculs utilisent principalement les CUDA Cores le mode tf32 permet d'utiliser les Tensor Cores spécialisés dans les calculs matriciels. Cela augmente fortement la vitesse de calcul : le débit passe de 717,9 à 1646,1 images/s, soit un gain de x2,29. La mémoire utilisée reste presque identique, car les données sont toujours stockées en 32 bits
- Réduction du trafic mémoire (bf16 et channels_last): Le format bf16 utilise 16 bits au lieu de 32 pour les activations. Les données prennent donc moins de place en mémoire et les transferts sont plus rapides. Le pic de mémoire passe de 5,93 à 3,29 Go, avec un gain de vitesse supplémentaire de x1,22. Le format channels_last organise ensuite les données de manière plus adaptée aux Tensor Cores. Cela évite certaines réorganisations inutiles et améliore les accés mémoire, apportant un gain supplémentaire de x1,67
- Fusion d'opérateurs (torch.compile): Normalement, PyTorch lance un noyau CUDA pour chaque opération, ce qui entraine plusieurs transferts de données entre la mémoire de GPU et les unités de calcul. torch.compile regroupe plusieurs opéations compatibles dans un même noyau. Les données intermédiaires sont ainsi conservées plus près des unités de calcul, ce qui réduit les transferts mémoire et les lancements de noyaux. Le pic mémoire descend à 3,07 Go et le débit gagne encore x1,08
Au total, la combinaison de ces optimisations permet d'obtenir un début environ 5x supérieur au fp32 de départ

&nbsp;

3.4) Les résultats confiment les estimations de l'ex 2:
- Avec 1 worker, le débit reste faible : 226 à 283 img/s en raw eet 330 à 499 img/s en small
- Avec 3 workers, le débt augmente, mais reste généralement inférieur au débit synthétique
- raw-3 avec compilation atteint 3 515,7 img/s grâce au cache RAM du système
En synthetic, le GPU est le principal facteur limitant. En raw et small, c'est surtout le chargement des images sur le CPU. La compilation n'apporte donc pas tout le temps de gain quand le chargement limite déjà les performances, elle peut même diminuer le débit comme elle utilise aussi des ressouces CPU, ce qui explique la loi Amdahl: accélérer le GPU ne suiffit pas si le CPU ne prépare pas les données rapidement

&nbsp;

3.5) Un GPU 1g.12gb a environ 1/7 de la puissance d'un H100 entier, soit 3 603/7 environ 515 img/s. Pour alimenter un H100 entier à 3 603 images/s, il faudrait environ 11 workers (3 603 / 341) pour raw et 9 workers (3 603 / 414) pour small. Un noeud arcadia disposant de 30 coeurs CPU peut largement supporter ce nombre de workers. In faudrait simplement demande + de CPU, avec par ex --cpus-per-task=12 ou 16

&nbsp;

3.6)La première compilation prend 49,6 sec alors que les suivantes prennent seulement entre 4,2 et 5,8 sec. cela explique par le cache de PyTorch : dans la 1ère exécution, les noyeux doivent être générés et compilés. Les executions suivantes réutilisent les fichiers déjà compilés
Le gain apporté par la compilation est d'environ 1,34 ms par itération. Il faut donc 36 itérations sans cache et 4 170 itérations avec cache. Pour 2 époques de 201 pas, soit 402 itérations, la compilation n'est donc pas rentable. Le temps économisé est juste d'environ 0,54 sec, contre 5,6 à 49,6 sec de compilation
torch.compile devient interessant surtout pour les entrainements longs.

&nbsp;

3.7) Le batch maximal atteint est de 2 200 images en bf16 et 1 128 images en fp32 donc le rapport est 2200 / 1128 soit environ 1,95. Le bacth maxest donc presque 2 fois plus grand en bf16. C'est cohérent avec le cours, car bf16 utilise 2 octzets par valeur contre 4 en fp32, la mémoire utilisée pas les activations est donc divisée par 2, le rapport n'est pas exactement égale à 2 car certains éléments comme les poids et l'états de l'optimiseur restent en fp32

&nbsp;

3.8)

| Époque | Temps | Débit | Précision top-1 |
| ----- | :---: | :---: | :---: |
| 0 | 48,4 s | 266 img/s | 99,0% |
| 1 | 12,3 s | 1 048 img/s | 99,6% |

La 1ère époque est plus longue car elle inclut environ 36,1 sec de compilation du modèle avec torch.compile et la 2e correspond davantage au temps réel d'entrainement. Par rapport à ref-raw, qui prenait 20,1 sec par époque, on obtient: 20,1 / 12,3 = 1,63. Le temps d'entrainement diminue donc

&nbsp;

## Exercice 4 : Entraînement multi-GPU avec DDP sous SLURM

&nbsp;

4.1) NCCL utilise TCP/IP via sockets sur l'interface Ethernet ens18, car aucun périphérique InfiniBand n'est disponible et le GPUDirect RDMA est désactivé. Pour l'all-reduce de 102 Mo:
- Durée : 877,2 ms
- Débit algbw : 0,12 Go/s
- Débit busbw : 0,12 Go/s
- Conversion : environ 0,93 Gbit/s
Cela correspond à un lien Ethernet 1 Gb/s

&nbsp;

4.2)
| Configuration | Temps/pas (ms) | Débit global (img/s) | Accélération S = D2 / D1 | Efficacité E = S/2 |
| ----- | :---: | :---: | :---: | :---: |
| l4-x1 | 136,9 | 467,6 | x1,00 | - |
| l4-x2 | 866,3 | 147,8 | x0,32 | 15,8 % |
| l4-x2-bf16hook | 463,4 | 276,2 | x0,59 | 29,5 % |
| l4-x1-accum8 | 1 036,9 | 493,8 | x1,00 | - |
| l4-x2-accum8-bf16hook | 1 358,7 | 753,6 | x1,53 | 76,3 % |

&nbsp;

4.3) Le modèle à 10 classes a 23,5 M paramètres, soit environ 94 Mo de gradients fp32
- Temps estimé de l'all-reduce : environ 807 ms
- Temps de calcul sur un GPU : 136,9 ms
- Sans recouvrement : 136,9 + 806,6 = 943,5 ms
- Avec recouvrement parfait : max(136,9;806,6) = 806,6 ms
- Temps mesuré sur deux noeuds : 866,3 ms
Le recouvrement a donc caché environ 77,2 ms de calcul pendant les communications

&nbsp;

4.4) Cette option convertit les gradients en bfloat16 avant leur envoi sur le réseau. Leur taille passe de 94 à environ 47 Mo, et ça réduit le temps de communication. Le temps par pas passe de 866,3 à 463,4 ms et le débit augmente de 147,8 à 276,2 images/s. Le risque est une perte de précision lors de la conversion ce qui peut rendre la convergeance moins stable ou réduire légèrement la précision finale

&nbsp;

4.5) Le loi d'Amdahl ne convient pas ici car elle suppose une charge fixe repartie entre plusieurs processeurs mais ici chaque GPU traite son propre batch et les communications peuvent être recouvertes par les calculs. Pour l4-x2-accum8-bf16hook, le calcul sur GPU avec accumulation 8 est de 1 036,9 ms, la communication en bf16 est de 806,6 / 2 = 403,3 ms et le temps recouvert de 77,2 ms. On prédit donc T = 1 036,9 + 403,3 - 77,2 = 1 363 ms. La mesure réelle est de 1 358,7 ms soit suelement 4,3 ms d'écart donc la prédiction est très proche

&nbsp;

4.6) Le DDP devient rentable quand le temps de communication non recouvert est inférieur au temps de calcul local. Il faut donc augmenter le rapport entre calcul et communication. 2 leviers possibles :
- Accumulation des gradients : plusieurs micro-batchs sont calculés avant une seule snchronisation. Prix : batch effectif plus grand et réglage du taux d'apprentissage plus compliqué
- Compression des gradients en bf16: réduit de moitié le volume transféré. Prix: perte possible de précision numérique et risque sur la convergence

&nbsp;

4.7) Le batch effectif passe de 64 à 128 puis 1 024. Il faudrait ajuster le lr pour un vrai entrainement:
- Avec AdamW: appliquer une régle en racine carrée. Multiplier le lr par la racine carrée de 2 pour 128 images puis par 4 pour 1 024 images. Ajouter warmup
- Avec SGD: appliquer une règle linéaire. Multiplier le lr oar 2 pour 128 images puis par 16 pour 1 024 images. Et là aussi ajouter un warmup

&nbsp;

4.8) Tous les processus utilisent la même graine pour l’initialisation car tous les GPU doivent commencer avec les mêmes poids pour entrainer le même modèle mais ils utilisent des données différentes car chaque GPU traite une partie différente du jeu de données afin de répartir le calcul et d'éviter de calculer plusieurs fois les mêmes gradients
Si on oublie sampler.set_epoch(epoch) dans la branche small, le mélange des données ne change pas entre les époques et les GPU revoient les mêmes images dans le même ordre et si on oublie model.no_sync() dans make_ddp_step, une synchronisation des gradients est lancée à chaque micro-batch au lieu d'une seule à la fin de l'accumulation et cela multiplie les communications et ralentit fortement l'entrainement

&nbsp;

## Exercice 5 : Quantification int8 sur CPU

&nbsp;

5.1)
| Variante | Taille | Latence | Premier appel | Top-1 |
| ----- | :---: | :---: | :---: | :---: |
| fp32-eager | 94,3 Mo | 49,85 ms | 0,1 s | 99,8 % |
| fp32 compilé | 94,3 Mo | 29,98 ms | 34,7 s | 99,8 % |
| fp32-eager | 47,2 Mo | 16,49 ms | 6,1 s | 99,8 % |
| fp32-eager | 23,9 Mo | 9,54 ms | 0,1 s | 99,8 % |

&nbsp;

5.2) Le rapport de taille entre fp32 et int8 vaut environ 3,95 et non exactement 4, ça s'explique par les métadonées de quantification et par certains tenseurs qui restent en fp32. weights_mb mesure uniquement les poids rééellement utilsés dans le graphe. Le state_dict contient aussi d'anciens poids fp32 conservés mais inutilisés après conversion, ce qui fausserait la mesure

&nbsp;

5.3) Les latences montrant que la compilation améloire déjà les performances : le temps passe de 49,85 ms en fp32 eager à 29,98 ms en fp32 compilé, soit un gain de x1,66. Ensuite le passage en bf16 réduit la latence à 16,49 ms (gain de x1,82) tandis que l'int8 atteint 9,54 ms soit un gain de x3,14 par rapport au fp32 compilé. Il faut comparer l'int8 au fp32 compilé plutôt qu'au fp32 eager, car les deux utilisent torch.compile. Cela permet d'isoler le gain apporté par la quantification, sans confondre avec celui de la compilation. Ce gain est notamment lié aux instructions matérielles VNNI et AMX du processeur qui accélèrent les calculs en int8

&nbsp;

5.4) Avec calib200, l’int8 conserve un top-1, un de 99,8 % un accord avec fp32 de 100 % et un écart moyen des logits de 0,0934. Le top-1 ne suffit pas : deux modèles peuvent prédire la même classe, tout en donnant des scores différents. L'écart des logits permet donc de mieux évaluer la fidélité de la quantification

| Calibration | Top-1 | Accord fp32 | Écat des logits |
| ----- | :---: | :---: | :---: |
| calib200 | 99,8 % | 100 % | 0,0934 |
| calib10 | 99,6 % | 99,8 % | 0,1007 |

&nbsp;

5.5)
| Variante | 1 thread | 4 threads | Accélération |
| ----- | :---: | :---: | :---: |
| fp32 compilé | 98,12 ms | 29,98 ms | x3,27 |
| int8 compilé | 29,46 ms | 9,54 ms | x3,09 |

L'accélération n'est pas proportionnelle au nombre de threads, elle reste inférieure à x4, notamment à cause des synchronisations et du partage des ressources CPU. Pour le déploiement, plusieurs threads réduisent la latence d'une requête, par contre plusieurs instance à un thread peuvent offrir un meilleur débit total lorsqque plusieurs requêtes arrivent en même temps

&nbsp;

5.6) La quantification dynamique concerne principalement les couches Linear et récurrentes mais pas les convolutions Conv2d. Or ResNet utilise surtout des convolutions : seule la dernière coucbe est linéaire et représente moins de 0,1 % des paramètres. La majorité du calcul reste donc en fp23. Le gain est rès faible et le calcul des plages des activations à chaque exécution ajoute même un coût supplémentaire

&nbsp;

## Rapport final

&nbsp;

### results_bench.csv

| tag | materiel | arch | precision | channels_last | compile | donnees | workers | pin_memory | batch | img_s | img_s_min | img_s_max | pic_memoire_go | compilation_s | epoque_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref-synth | NVIDIA H100 NVL | resnet50 | fp32 | False | False | synthetic |  |  | 64 | 719.5 | 717.3 | 721.6 | 5.95 | nan | nan |
| ref-raw | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 3 | True | 64 | 707.4 | 705.4 | 709.5 | 5.95 | nan | 20.1 |
| raw-w0 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 0 | True | 64 | 152.7 | 148.7 | 156.8 | 5.95 | nan | nan |
| raw-w1 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 1 | True | 64 | 324.5 | 322.9 | 326.1 | 5.95 | nan | nan |
| raw-w2 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 2 | True | 64 | 692.8 | 692.7 | 692.9 | 5.95 | nan | nan |
| raw-w3 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 3 | True | 64 | 684.8 | 681.7 | 688.0 | 5.95 | nan | nan |
| raw-w3-nopin | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 3 | False | 64 | 617.2 | 610.8 | 623.7 | 5.95 | nan | nan |
| small-w0 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | small | 0 | True | 64 | 260.8 | 260.0 | 261.7 | 5.95 | nan | nan |
| small-w3 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | small | 3 | True | 64 | 679.8 | 678.3 | 681.2 | 5.95 | nan | nan |
| raw-w0 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 0 | True | 64 | 178.9 | 175.9 | 181.9 | 5.95 | nan | nan |
| raw-w0 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 0 | True | 64 | 231.2 | 229.8 | 232.5 | 5.95 | nan | nan |
| raw-w1 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 1 | True | 64 | 342.8 | 339.6 | 346.0 | 5.95 | nan | nan |
| raw-w2 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 2 | True | 64 | 701.2 | 695.4 | 707.0 | 5.95 | nan | nan |
| raw-w3 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 3 | True | 64 | 699.4 | 696.8 | 701.9 | 5.95 | nan | nan |
| raw-w3-nopin | NVIDIA H100 NVL | resnet50 | fp32 | False | False | raw | 3 | False | 64 | 619.1 | 615.0 | 623.3 | 5.95 | nan | nan |
| small-w0 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | small | 0 | True | 64 | 262.7 | 262.1 | 263.2 | 5.95 | nan | nan |
| small-w3 | NVIDIA H100 NVL | resnet50 | fp32 | False | False | small | 3 | True | 64 | 690.2 | 690.0 | 690.5 | 5.95 | nan | nan |
| fp32-synthetic | NVIDIA H100 NVL | resnet50 | fp32 | False | False | synthetic |  |  | 64 | 717.9 | 716.7 | 718.4 | 5.95 | nan | nan |
| tf32-synthetic | NVIDIA H100 NVL | resnet50 | tf32 | False | False | synthetic |  |  | 64 | 1646.1 | 1629.5 | 1672.7 | 5.93 | nan | nan |
| bf16-synthetic | NVIDIA H100 NVL | resnet50 | bf16 | False | False | synthetic |  |  | 64 | 2008.1 | 2007.8 | 2008.1 | 3.29 | nan | nan |
| bf16-cl-synthetic | NVIDIA H100 NVL | resnet50 | bf16 | True | False | synthetic |  |  | 64 | 3349.8 | 3246.4 | 3413.9 | 3.29 | nan | nan |
| bf16-cl-compile-synthetic | NVIDIA H100 NVL | resnet50 | bf16 | True | True | synthetic |  |  | 64 | 3602.9 | 3570.6 | 3940.3 | 3.07 | 49.6 | nan |
| bf16-cl-raw-w1 | NVIDIA H100 NVL | resnet50 | bf16 | True | False | raw | 1 | True | 64 | 282.6 | 261.1 | 298.1 | 3.29 | nan | nan |
| bf16-cl-compile-raw-w1 | NVIDIA H100 NVL | resnet50 | bf16 | True | True | raw | 1 | True | 64 | 226.5 | 205.0 | 232.0 | 3.07 | 5.8 | nan |
| bf16-cl-raw-w3 | NVIDIA H100 NVL | resnet50 | bf16 | True | False | raw | 3 | True | 64 | 1404.5 | 715.7 | 2505.3 | 3.29 | nan | nan |
| bf16-cl-compile-raw-w3 | NVIDIA H100 NVL | resnet50 | bf16 | True | True | raw | 3 | True | 64 | 3515.7 | 3436.7 | 3524.1 | 3.07 | 5.6 | nan |
| bf16-cl-small-w1 | NVIDIA H100 NVL | resnet50 | bf16 | True | False | small | 1 | True | 64 | 498.6 | 487.3 | 500.4 | 3.29 | nan | nan |
| bf16-cl-compile-small-w1 | NVIDIA H100 NVL | resnet50 | bf16 | True | True | small | 1 | True | 64 | 330.0 | 326.5 | 381.8 | 3.07 | 4.2 | nan |
| bf16-cl-small-w3 | NVIDIA H100 NVL | resnet50 | bf16 | True | False | small | 3 | True | 64 | 2673.2 | 2623.3 | 2686.6 | 3.29 | nan | nan |
| bf16-cl-compile-small-w3 | NVIDIA H100 NVL | resnet50 | bf16 | True | True | small | 3 | True | 64 | 1387.7 | 1106.5 | 2809.2 | 3.07 | 5.5 | nan |

### results_ddp.csv

| tag | materiel | processus | noeuds | backend | precision | bf16_hook | donnees | accumulation | batch_par_processus | batch_effectif | pas_ms | img_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l4-x1 | NVIDIA L4 | 1 | 1 | nccl | bf16 | False | synthetic | 1 | 64 | 64 | 136.9 | 467.6 |
| l4-x2 | NVIDIA L4 | 2 | 2 | nccl | bf16 | False | synthetic | 1 | 64 | 128 | 866.3 | 147.8 |
| l4-x2-bf16hook | NVIDIA L4 | 2 | 2 | nccl | bf16 | True | synthetic | 1 | 64 | 128 | 463.4 | 276.2 |
| l4-x1-accum8 | NVIDIA L4 | 1 | 1 | nccl | bf16 | False | synthetic | 8 | 64 | 512 | 1036.9 | 493.8 |
| l4-x2-accum8-bf16hook | NVIDIA L4 | 2 | 2 | nccl | bf16 | True | synthetic | 8 | 64 | 1024 | 1358.7 | 753.6 |

### results_maxbatch.csv

| tag | materiel | precision | channels_last | batch_max |
|---|---|---|---|---|
| ref-maxbatch | NVIDIA H100 NVL | fp32 | False | 1128 |
| maxbatch-bf16 | NVIDIA H100 NVL | bf16 | True | 2200 |

### results_quant.csv

| tag | variante | threads | calibration | taille_mo | latence_ms | premier_appel_s | top1_val | accord_fp32 | ecart_logits | n_val |
|---|---|---|---|---|---|---|---|---|---|---|
| calib200 | fp32-eager | 4 |  | 94.3 | 49.85 | 0.1 | 0.998 | 1.0 | 0.0 | 500 |
| calib200 | fp32 | 4 |  | 94.3 | 29.98 | 34.7 | 0.998 | 1.0 | 0.0 | 500 |
| calib200 | bf16 | 4 |  | 47.2 | 16.49 | 6.1 | 0.998 | 1.0 | 0.0229 | 500 |
| calib200 | int8 | 4 | 200 | 23.9 | 9.54 | 4.6 | 0.998 | 1.0 | 0.0934 | 500 |
| calib10 | int8 | 4 | 10 | 23.9 | 10.96 | 13.8 | 0.996 | 0.998 | 0.1007 | 500 |
| threads1 | fp32 | 1 |  | 94.3 | 98.12 | 13.8 | 1.0 | 1.0 | 0.0 | 100 |
| threads1 | int8 | 1 | 200 | 23.9 | 29.46 | 3.1 | 0.99 | 0.99 | 0.0994 | 100 |

### results_train.csv

| tag | materiel | epoque | precision | channels_last | compile | donnees | workers | epoque_s | img_s | top1_val |
|---|---|---|---|---|---|---|---|---|---|---|
| best | NVIDIA H100 NVL | 0 | bf16 | True | True | small | 3 | 48.4 | 265.5 | 0.99 |
| best | NVIDIA H100 NVL | 1 | bf16 | True | True | small | 3 | 12.3 | 1047.6 | 0.996 |

&nbsp;

### Questions

&nbsp;

1) 
| Configuration | Matériel | Données | Débit (img/s) | Pic mémoire / taille | Temps/époque |
| ----- | :---: | :---: | :---: | :---: | :---: |
| ref-synth fp32 | H100 NVL | Synthétiques | 719,5 | 5,95 Go | - |
| ref-raw fp32 | H100 NVL | Raw, 3 workers | 707,4 | 5,95 Go | 20,1 s |
| ref-w0 | H100 NVL | Raw, 0 worker | 231,2 | 5,95 Go | - |
| ref-w1 | H100 NVL | Raw, 1 worker | 342,8 | 5,95 Go | - |
| small-w0 | H100 NVL | Small, 0 worker | 262,7 | 5,95 Go | - |
| small-w3 | H100 NVL | Small, 3 workers | 690,2 | 5,95 Go | - |
| tf32-synthetic | H100 NVL | Synthétiques | 1 646,1 | 5,93 Go | - |
| bf16-synthetic | H100 NVL | Synthétiques | 2 008,1 | 3,29 Go | - |
| bf16-cl-synthetic | H100 NVL | Synthétiques | 3 349,8 | 3,29 Go | - |
| bf16-cl-compile | H100 NVL | Synthétiques | 3 602,9 | 3,07 Go | - |
| best | H100 NVL | Small, 3 workers | 1 047,6 | 3,07 Go | 12,3 s (48,4 s à la 1re époque) |
| l4-x1 | 1 L4 | Synthétiques | 467,6 | - | - |
| l4-x2 | 2 L4, 2 noeuds | Synthétiques | 147,8 | - | - |
| l4-x2-bf16hook | 2 L4, 2 noeuds | Synthétiques | 276,2 | - | - |
| l4-x2-accum8-bf16hook | 2 L4, 2 noeuds | Synthétiques | 753,6 | - | - |
| fp32-eager CPU | CPU, 4 threads | Small, validation | 20,1 | 94,3 Mo | - |
| fp32 compilé CPU | CPU, 4 threads | Small, validation | 33,4 | 94,3 Mo | - |
| bf16 compilé CPU | CPU, 4 threads | Small, validation | 60,6 | 47,2 Mo | - |
| int8 compilé CPU | CPU, 4 threads | Small, validation | 104,8 | 23,9 Mo | - |

&nbsp;

2) Avec deux GPU L4, le débit idéal serait de 935,2 images/s soit deux fois les 467,6 images/s obtenues avec un seul GPU. Mais, l4-x2 atteint seulement 147,8 images/s. La principale cause est la communication réseau : les deux noeuds un Ethernet 1 Gb/s. L'all–reduce de 94 Mo prend environ 806,6 ms, contre seulement 136,9 ms de calcul sur un GPU. Même avec 77,2 ms de communication masquée par le calcul, il reste 729,4 ms  d'attente réseau. Le bf16–hook, réduit le temps de communication à environ 403,3 ms. Avec l'accumulation sur 8 micro-batchs, le débit monte à 753,6 images/s mais il reste inférieur au débit idéal. Le chargement des données ne joue aucun rôle dans ces mesures DDP car elles utilisent des données synthétiques. Enfin la synchronisation impose d'attendre le processus le plus lent. Les différences de temps de calcul ou de chargement entre les nœuds peuvent donc ralentir l'ensemble, même si cet effet est négligeable dans le benchmark synthétique.

&nbsp;

3)
- Pour entraîner le modèle sur le cluster: Je choisirai un seul GPU H100 NVL, avec bf16 channels_last et des données pré-redimensionnées (small) chargées avec plusieurs workers et pin_memory=True. Cette configuration évite les communications coûteuses entre les nœuds et permet d'exploiter efficacement le GPU. La compilation avec torch.compile est intéressante pour un entraînement long car elle augmente encore le débit jusqu'à 3602,9 image/s sur données synthétique. Par contre, pour un entraînement court, son temps de compilation initial peut annuler le gain. Je choisirais pas le DDP multi-noeuds sur ce réseau Ethernet 1 Gb/s : même avec la compression bf16 et l'accumulation, il reste pénalisé par les communications
- Pour déployer le modèle sur CPU: Je recommande la variante int8 compilée avec une calibration sur 200 images. Elle réduit la taille du modèle de 94,3 à 23,9 Mo et atteint une latence de 9,54 ms par image sur 4 threads, tout en conservant un top-1 de 99,8 % et un accord de 100 % avec fp32. Pour une requête isolée, 4 threads permettent de réduire la latence. Pour traiter plusieurs requêtes simultanément, plusieurs instances indépendantes à un thread peuvent offrir un meilleur débit global

