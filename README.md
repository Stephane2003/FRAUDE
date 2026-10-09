# Détection de fraude en temps réel — Pipeline Streaming et Machine Learning

## Présentation du projet

Ce projet consiste à concevoir un pipeline de traitement de transactions permettant de détecter des comportements potentiellement frauduleux à l'aide du Machine Learning.

L'objectif est de combiner le traitement de données en streaming, des modèles de détection d'anomalies et des outils de visualisation afin de faciliter l'identification et le suivi des transactions suspectes.

Le projet s'appuie sur un jeu de données de transactions bancaires et met en œuvre une architecture orientée données, conteneurisée avec Docker.

## Objectifs

* Mettre en place un pipeline de traitement de transactions avec Apache Kafka.
* Développer et évaluer des modèles de Machine Learning pour la détection de fraude.
* Explorer les approches de détection d'anomalies et de classification supervisée.
* Centraliser les données et les résultats de détection avec Elasticsearch.
* Visualiser les informations utiles au suivi des fraudes avec Kibana.
* Utiliser Docker pour faciliter le déploiement de l'environnement technique.

## Architecture technique

Le projet s'organise autour des composants suivants :

| Composant               | Rôle                                                        |
| ----------------------- | ----------------------------------------------------------- |
| Python                  | Préparation des données et développement des modèles        |
| Apache Kafka            | Transmission des transactions dans le pipeline de streaming |
| Isolation Forest        | Détection non supervisée des anomalies                      |
| XGBoost                 | Classification des transactions frauduleuses                |
| Elasticsearch           | Indexation et recherche des transactions et résultats       |
| Kibana                  | Visualisation et exploration des données                    |
| Docker / Docker Compose | Conteneurisation et orchestration des services              |

### Fonctionnement général

1. **Acquisition des données :** chargement et préparation des transactions.
2. **Streaming :** transmission des événements à travers Apache Kafka.
3. **Détection :** utilisation des modèles de Machine Learning pour identifier les transactions suspectes.
4. **Centralisation :** indexation des données et résultats dans Elasticsearch.
5. **Visualisation :** exploration des résultats à l'aide de Kibana.

## Modèles de Machine Learning

Deux approches ont été étudiées :

### Isolation Forest

Algorithme de détection non supervisée permettant d'identifier des observations atypiques à partir des caractéristiques des transactions.

* Type : détection d'anomalies.
* Objectif : repérer les transactions présentant des comportements inhabituels.
* Résultat expérimental : AUC de **0,9455**.

### XGBoost

Algorithme de classification supervisée utilisé pour distinguer les transactions normales des transactions frauduleuses.

Résultats expérimentaux obtenus sur le jeu de test :

| Métrique          |  Résultat |
| ----------------- | --------: |
| AUC ROC           |    0,9673 |
| Accuracy          |   99,95 % |
| Précision         |   89,71 % |
| Rappel            |   82,43 % |
| F1-score          |   85,92 % |
| Fraudes détectées | 68 sur 74 |

Ces résultats correspondent aux expérimentations réalisées sur le jeu de données utilisé. Ils ne constituent pas une garantie de performance en production.

**Remarque :** dans un problème de détection de fraude, l'accuracy seule est insuffisante en raison du déséquilibre entre les transactions normales et frauduleuses. La précision, le rappel, le F1-score et l'AUC apportent des informations complémentaires sur la qualité du modèle.

## Jeu de données

Le projet utilise le jeu de données public **Credit Card Fraud Detection**, disponible sur Kaggle.

Ce jeu de données contient des transactions par carte bancaire, avec une variable cible indiquant si une transaction est frauduleuse.

Lien : [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Le jeu de données original contient 284 807 transactions, dont 492 transactions frauduleuses. Il présente donc un fort déséquilibre entre les classes.

Les données sont destinées à l'expérimentation et à l'évaluation des modèles de détection de fraude.

## Installation et exécution

### Prérequis

* Git
* Docker Desktop
* Docker Compose
* Python 3 et les dépendances nécessaires au projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/Stephane2003/FRAUDE.git
cd FRAUDE
```

### 2. Lancer les services Docker

Si le dépôt contient un fichier `docker-compose.yml` ou `compose.yaml` configuré pour l'ensemble du pipeline :

```bash
docker compose up --build -d
```

### 3. Vérifier les conteneurs

```bash
docker compose ps
```

### 4. Consulter les journaux

```bash
docker compose logs -f
```

Pour arrêter les services :

```bash
docker compose down
```



## Organisation du projet

Le dépôt regroupe les éléments du pipeline de détection de fraude, notamment :

* Les scripts de préparation et de traitement des données.
* Les composants de streaming Kafka.
* Les modèles et scripts de Machine Learning.
* Les configurations des services Docker.
* Les composants de stockage et de visualisation des résultats.

La structure exacte dépend de l'organisation des fichiers dans le dépôt.

## Sécurité et limites

Ce projet est un projet expérimental à vocation académique et technique.

* Les données utilisées sont destinées à l'expérimentation.
* Les résultats des modèles dépendent du jeu de données et du protocole d'évaluation.
* Une utilisation réelle nécessiterait notamment une surveillance de la dérive des données, une évaluation sur des données récentes et des mécanismes de contrôle des faux positifs.
* Aucune donnée bancaire personnelle ni aucun secret d'accès ne doit être ajouté au dépôt GitHub.

## Perspectives d'amélioration

* Ajouter un suivi des performances des modèles en continu.
* Mettre en place une gestion des seuils de détection configurable.
* Renforcer la supervision du pipeline et la gestion des erreurs.
* Ajouter des tests automatisés et une intégration continue (CI/CD).
* Évaluer les modèles sur des données plus récentes afin de mesurer leur capacité de généralisation.

##  Auteur

**Ola PRODJINOTHO**

Étudiant en Master 1 — Systèmes d'Information et Data Science

Centres d'intérêt : Data Science, Machine Learning, Data Engineering et intelligence artificielle.

