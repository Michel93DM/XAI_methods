# Exploration et application des méthodes d'explicabilité pour l'amélioration des modèles de Machine Learning

Ce projet est realisé dans le cadre du Matser Spécialisé IA de confiance a CentralSupElec.

## Datasets

1. **Dataset CyberSecurity**

2. **Oxford Parkinson's Disease Telemonitoring Dataset**
  - ref: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

Each Dataset is described in its corresponding Notebook.

## Repository structure

-   data: our data, to allow you to re-execute the notebooks.
    - dataset_CyeberSecruity
    - dataset_Parkinson

-   notebooks: one notebook for each data source, testing the different Xplainability methods.
    - CyberSecurity
    - Parkinson_telemonitoring

-   src: our Python package with plot functions for the different Xplainability methods.

## Xaplainable methods  
In this work, we present the methods: 

- Global: 
  - Shap
  - PDP
  - ALE
  - LOFO

- Local:
  - Lime
  - ICE
  - Anchors
  - Shap 