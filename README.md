# Exploration et application des méthodes d'explicabilité pour l'amélioration des modèles de Machine Learning

Ce projet est realisé dans le cadre du Matser Spécialisé IA de confiance a CentralSupElec.

## TODO

- [x] Explications des jeux de données (dans le rapport)
  - [x] Données du challenge -> Appeler ça autrement 
  - [x] Parkinson telemonitoring
- [ ] Remplir le rapport
  - [ ] SHAP
    - [x] Explications théoriques
    - [x] Données challenge
    - [x] Données Parkinson
    - [ ] Tester les autres types de graphiques (decision plot et dependance plot)
  - [ ] PDP
    - [x] Explications théoriques
    - [x] Données challenge
    - [x] Données Parkinson
  - [ ] ALE
    - [x] Explications théoriques
    - [x] Données challenge
    - [ ] Données Parkinson
  - [ ] LOFO
    - [x] Explications théoriques
    - [x] Données challenge
    - [ ] Données Parkinson
  - [ ] ICE
    - [x] Explications théoriques
    - [ ] Données challenge
    - [x] Données Parkinson
  - [ ] Anchors
    - [x] Explications théoriques
    - [ ] Données challenge
    - [x] Données Parkinson
  - [ ] LIME
    - [x] Explications théoriques
    - [ ] Données challenge
    - [x] Données Parkinson
- [x] Faire un tableau récapitulatif des méthodes. Comparer local, global, rapidité, facilité d'implémentation, lister les packages Python, l'applicabilité à différents types de modèles (deep, sklearn + pipeline, autres)
- [ ] Nettoyer notebooks (mettre fonction dans des fichiers Python + écrire analyses)
    - [ ] Données challenge
    - [ ] Données Parkinson
- [ ] Est-ce qu'on ajoute aussi le package interpretml ? Oui

## Datasets used

1. **Dataset from the Sorbone Data Challenge**

(Description)

2. **Oxford Parkinson's Disease Telemonitoring Dataset**

https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

## Repository structure

-   data: our data, to allow you to re-execute the notebooks.

-   notebooks: one notebook for each data source, testing the different Xplainability methods.

-   src: our Python package with plot functions for the different Xplainability methods.


## Xplainability methods

### SHAP

La méthode **SHAP** (SHapley Additive exPlanations) est une approche d'interprétabilité des modèles d'apprentissage automatique basée sur la théorie des jeux coopératifs. Elle attribue à chaque caractéristique (feature) une valeur d'importance, appelée *valeur de Shapley*, qui reflète sa contribution à la prédiction du modèle.

#### Objectif

L'objectif principal de SHAP est d'**expliquer la prédiction d'un modèle pour une instance $x$** donnée en la décomposant comme une **somme de contributions individuelles de chaque caractéristique**.  
SHAP calcule les valeurs de Shapley en considérant les valeurs des caractéristiques comme des joueurs dans une coalition, et la prédiction comme un gain à répartir équitablement entre eux.

Le modèle explicatif est défini comme suit :

$$
g(z') = \phi_0 + \sum_{j=1}^{M} \phi_j z'_j
$$

où :

- $g$ est le modèle explicatif,
- $z' \in \{0,1\}^M$ est le **vecteur de coalition** (indiquant la présence ou l’absence d’une caractéristique),
- $M$ est le nombre total de caractéristiques,
- $\phi_j$ est la valeur de Shapley pour la caractéristique $j$.

Si toutes les caractéristiques sont présentes dans la coalition ($z'$ est un vecteur de 1), l’équation devient :

$$
g(x') = \phi_0 + \sum_{j=1}^{M} \phi_j
$$

L'innovation majeure de cette approche est d'exprimer les valeurs de Shapley sous forme de **modèle additif linéaire**, ce qui les rend compatibles avec des méthodes locales d'explication.

#### Les valeurs de Shapley

Les **valeurs de Shapley** proviennent de la théorie des jeux coopératifs. L’idée est d’estimer la **contribution marginale moyenne** d’une caractéristique à toutes les coalitions possibles.

Formellement, la valeur de Shapley pour la caractéristique $i$ est définie comme :

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \cdot \left[f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)\right]
$$

où :

* $N$ est l'ensemble des caractéristiques,
* $S$ est un sous-ensemble de $N$ ne contenant pas $i$,
* $f_S(x_S)$ est la prédiction du modèle en ne considérant que les caractéristiques dans $S$.

#### Avantages

Les valeurs de Shapley possèdent plusieurs **propriétés théoriques désirables** :

* **Efficience** : La somme des contributions est égale à la différence entre la prédiction et la valeur de base.
* **Symétrie** : Si deux caractéristiques contribuent de manière identique, elles obtiennent la même valeur.
* **Nullité** : Si une caractéristique n’affecte jamais la prédiction, sa valeur SHAP est nulle.
* **Additivité** : Pour deux modèles combinés, les valeurs SHAP se combinent également.

#### Limites

Le calcul exact des valeurs de Shapley est **combinatoire** : il nécessite d’évaluer toutes les coalitions possibles, ce qui engendre une complexité exponentielle $O(2^M)$. Cela rend le calcul intractable pour des modèles comportant de nombreuses variables. Pour pallier ce problème, plusieurs **approximations** ont été proposées.


### Les différentes variantes de SHAP

Plusieurs méthodes ont été développées pour adapter SHAP à différents types de modèles :

* **KernelSHAP** :

  * Méthode générale, applicable à tout modèle (black-box).
  * Basée sur une régression localement pondérée.
  * Approximative mais très flexible.

* **TreeSHAP** :

  * Spécialisée pour les modèles d’arbres (comme XGBoost, LightGBM, CatBoost).
  * Permet un **calcul exact et rapide** des valeurs de Shapley.
  * Exploite la structure arborescente pour éviter l’énumération explicite des coalitions.

* **DeepSHAP** :

  * Conçu pour les réseaux de neurones.
  * Combine des idées de SHAP avec DeepLIFT pour approximer les valeurs de Shapley.
  * Nécessite certaines hypothèses sur l’architecture (comme la différentiabilité).

* **LinearSHAP** :

  * Méthode exacte pour les modèles linéaires.
  * Très rapide et peu coûteuse computationnellement.

* **PartitionExplainer** :

  * Variante pour les modèles d’arbres avec interactions complexes.
  * Utilise une stratégie de partitionnement récursif de l’arbre.

* **SamplingSHAP** :

  * Version échantillonnée de KernelSHAP pour les cas avec beaucoup de variables.
  * Approximative mais plus scalable.

### Les packages Python

Le package principal est :

* **`shap`** :
  Bibliothèque officielle développée par Scott Lundberg (auteur principal de SHAP). Elle permet de :

  * Calculer les valeurs SHAP avec toutes les méthodes mentionnées (kernel, tree, deep, linear).
  * Générer des visualisations :

    * **summary plot** (vue d’ensemble),
    * **dependence plot** (relation entre une variable et sa valeur SHAP),
    * **force plot** (visualisation locale),
    * **decision plot** (chemin de décision cumulatif).

Exemple d’utilisation :

```python
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---

### LIME


-   ICE
-   Anchors


### PDP – Partial Dependence Plot

Le **Partial Dependence Plot (PDP)** est une méthode d’interprétabilité **globale** permettant de visualiser la relation entre une ou plusieurs variables d’entrée et la sortie d’un modèle d’apprentissage automatique. Il répond à la question :

> *« Quelle est l'influence moyenne d'une variable sur la prédiction du modèle ? »*

Le PDP estime **l’effet marginal moyen** d’une ou plusieurs caractéristiques $x_S \subset x$ sur la prédiction $f(x)$, en **marginalisant** les autres variables $x_C = x \setminus x_S$.

#### Définition formelle

La **fonction de dépendance partielle** est définie comme :

$$
\hat{f}_S(x_S) = \mathbb{E}_{x_C}[\hat{f}(x_S, X_C)]
$$

où :

- $S$ est le sous-ensemble de caractéristiques d’intérêt,
- $C$ est le complément de $S$,
- $\hat{f}_S(x_S)$ représente la prédiction moyenne lorsque $x_S$ est fixé et $x_C$ varie selon sa distribution dans les données.

#### Méthode de calcul (approximation empirique)

En pratique, cette espérance est approximée à partir du jeu de données $\{x^{(i)}\}_{i=1}^n$ :

$$
\hat{f}_S(x_S) \approx \frac{1}{n} \sum_{i=1}^{n} f(x_S, x_C^{(i)})
$$

Pour chaque valeur (ou grille de valeurs) de $x_S$, on crée des **instances artificielles** en combinant cette valeur avec tous les $x_C^{(i)}$ observés, puis on calcule la moyenne des prédictions du modèle.

#### Applications

Le PDP est principalement utilisé pour :

* **Comprendre les effets moyens** d’une variable sur la prédiction.
* **Visualiser la forme fonctionnelle** : linéaire, monotone, en U, etc.
* **Détecter des interactions** (avec PDP bivarié).
* **Analyser des modèles complexes** comme les forêts aléatoires, les boosting trees ou les réseaux de neurones.

#### Limites

Malgré son utilité, le PDP présente plusieurs **limitations importantes** :

* **Hypothèse d’indépendance** :
  Il suppose que les variables $x_S$ et $x_C$ sont **indépendantes**. En présence de **corrélations fortes**, les combinaisons générées peuvent être **non réalistes**, voire **inexistantes** dans les données.

* **Effets locaux masqués** :
  Le PDP **moyenne** les prédictions sur tout le jeu de données, ce qui peut **masquer des comportements locaux** ou des **interactions complexes**.
  Pour remédier à cela, on utilise les **ICE plots** (Individual Conditional Expectation), qui montrent les courbes individuelles pour chaque instance. ICE est donc complémentaire à PDP, et plus adaptée à l’analyse **locale**.

---

### PDP et importance des variables

En 2018, Greenwell, Boehmke et McCarthy ont proposé une métrique appelée **PD-based Feature Importance**, qui consiste à mesurer la **variabilité du PDP** d'une caractéristique. Cette mesure peut servir comme **indicateur d’importance globale**.
Cependant, elle repose elle aussi sur la **supposition d’indépendance** des variables, et souffre donc des mêmes limites.

Une alternative plus robuste à cette hypothèse est l’**Accumulated Local Effects (ALE)**, qui se base sur la **distribution conditionnelle** plutôt que marginale. ALE est présenté dans la section suivante.

---

### Librairies Python

Les PDP peuvent être facilement générés avec les bibliothèques suivantes :

* **`scikit-learn`** :

  * Fonction : `sklearn.inspection.PartialDependenceDisplay.from_estimator`.
  * Supporte les modèles de type `sklearn`.

  ```python
  from sklearn.inspection import PartialDependenceDisplay
  PartialDependenceDisplay.from_estimator(model, X, features=[0])
  ```

* **`pdpbox`** :

  * Outil spécialisé développé par le Dr. Terence Parr.
  * Permet une plus grande personnalisation (PDP 1D, 2D, ICE inclus).

* **`interpret`** (par Microsoft) :

  * Propose PDP et autres méthodes globales avec visualisations interactives.

* **`DALEX` (via Python ou R)** :

  * Fournit des PDP, ICE, ALE dans un cadre uniforme pour l’interprétabilité.

---

### ALE – Accumulated Local Effects

L’**Accumulated Local Effects (ALE)** est une méthode d’interprétabilité **globale** conçue pour analyser l’effet d’une caractéristique d’entrée sur la prédiction d’un modèle tout en étant **robuste aux corrélations** entre variables. Contrairement au **Partial Dependence Plot (PDP)**, qui repose sur une **marginalisation** naïve, ALE utilise la **distribution conditionnelle**, évitant ainsi les combinaisons irréalistes de variables.

Elle répond à la question :

> *« Quel est l'effet moyen local d'une variable sur la prédiction, en tenant compte de la distribution réelle des données ? »*

---

### Intuition

* Le PDP calcule l’effet moyen **en remplaçant** une variable dans toutes les observations — sans tenir compte des relations avec les autres.
* ALE, lui, estime les **variations locales du modèle** dans des **zones de données réellement observées**, puis **accumule** ces effets.

---

### Méthode de calcul

Soit $x_j$ la variable que l'on souhaite interpréter. La procédure ALE se déroule en quatre étapes principales :

1. **Partitionnement**
   
   Le domaine de $x_j$ est divisé en $K$ intervalles $[z_{k-1}, z_k]$, souvent définis à partir des **quantiles** des données (bins équi-fréquentiels).

2. **Effets locaux**
   
   Pour chaque intervalle $[z_{k-1}, z_k]$, on estime la variation moyenne du modèle :

   $$
   \Delta f_j(z_k) = \mathbb{E}_{\mathbf{x}_{\setminus j} \mid x_j \in [z_{k-1}, z_k]} \left[ f(z_k, \mathbf{x}_{\setminus j}) - f(z_{k-1}, \mathbf{x}_{\setminus j}) \right]
   $$

   Cela correspond à la **variation locale moyenne** dans chaque bin.

3. **Effets accumulés**
   
   Les effets sont ensuite **intégrés (accumulés)** à partir de la borne inférieure :

   $$
   ALE_j(z_k) = \sum_{i=1}^{k} \Delta f_j(z_i)
   $$

4. **Centrage**
   Pour rendre l’effet **interprétable** (comparaison entre variables), la fonction est centrée autour de zéro :

   $$
   \tilde{ALE}_j(z_k) = ALE_j(z_k) - \frac{1}{K} \sum_{i=1}^{K} ALE_j(z_i)
   $$

---

### Formule continue (interprétation dérivée)

L’approche peut être vue comme une **intégration de la dérivée partielle moyenne** du modèle :

$$
\hat{f}_{j, ALE}(x_j) = \int_{z_0}^{x_j} \mathbb{E}_{X_C \mid X_j = z} \left[ \frac{\partial \hat{f}(X_j, X_C)}{\partial X_j} \right] dz - \text{constante}
$$

où :

* $\hat{f}$ est le modèle de prédiction,
* $X_C$ représente toutes les autres variables,
* La dérivée mesure **l'effet local instantané** de $X_j$.

---

### Avantages

- **Robuste aux corrélations** entre variables (contrairement au PDP).
- **Respecte la distribution réelle des données**.
- **Plus rapide à calculer** que SHAP pour les grands modèles.
- Permet une **interprétation locale agrégée** (effets locaux → vue globale).
- Facilement visualisable (graphique 2D, parfois 3D pour les interactions).

---

### Inconvénients

- Ne fournit pas d’attribution individuelle (contrairement à SHAP).
- Les effets **ne sont pas additifs** (on ne peut pas sommer les contributions).
- La qualité dépend du **choix du nombre de bins** (partitionnement).
- Méthode encore **peu connue et moins diffusée** que SHAP ou PDP.

---

### Librairies Python

Plusieurs bibliothèques permettent de générer des graphiques ALE :

* **`interpret`** (Microsoft) :

  * Implémente ALE dans un cadre interactif et unifié.
  * Permet d’utiliser ALE avec des modèles scikit-learn ou LightGBM.

* **`pyALE`** (très complet) :

  * Calcul ALE 1D ou 2D pour détecter des interactions.
  * Supporte nativement les pipelines scikit-learn.

  ```python
  from pyALE import ale
  ale_eff = ale(X=X, model=clf, feature=['feature_name'], include_CI=True)
  ```

* **`alibi`**

---

-   LOFO
