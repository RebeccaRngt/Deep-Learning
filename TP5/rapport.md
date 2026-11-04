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

## Exercice 3:  Module d'Attention et LSTM Personnalisé

&nbsp;

3.a) La variable context_vector renvoyé par le module correspond conceptuellement à une combinaison pondérée des caractéristiques visuelles des 49 régions de l'image, sélectonnées selon l'imoortance attibuée par me mécanisme d'attention. Il résout le problème du goulot d'étrangement informationnel du Seq2Seq classique qui compresse toute l'image en un seul vecteur fixe, en permettant au RNN de se concentrer sur différentes régions de l'image à chaque étape de génération

&nbsp;

## Exercice 4: Modèle Global et Boucle d'Entraînement

&nbsp;

[Courbes TensorBoard Learning rate](images/courbe_lr.png)

&nbsp;

[Courbes TensorBoard Loss](images/courbe_loss.png)

&nbsp;

4.2) 
- Epoque 1: two men are sitting in in front with with men are in man in other man in on be
    la légende contient beaucoup de répétitions et présente une synthaxxe peu cohérente.
- Epoque 10: two men in in diving in the pool with them are in is in other is is is be
    Le modèle utilise des mots plus pertinents par rapport à l'image, comme diving et pool. Cependant, les répétitions persistent et les phrases restent grammaticalement incorrectes

&nbsp;

4.3) Si on gardait le lr=0.001 constant au lieu de le réduire (tous les 5 epoques avec le facteurb 0,1 dans notre code), les mises à jour des poids pourraient rester trop importantes en fin d'entrainement et provoquer des oscillations et une convergence moins stable
Avec Adam ou RMSProp, la décroissance du LR est particulièrement pertinente car meêl si ces optimiseurs adaptent les mises à jour grâce aux gradients précédents, réduire le learning global global permet d'effectuer des ajustements plus fins en fin d'entrainement

&nbsp;
&nbsp;

##### Notes: Le fichier train.py s'appelle train_TP5 sur le cluster car un fichier train.py existent déjà (crée lors du TP1)
