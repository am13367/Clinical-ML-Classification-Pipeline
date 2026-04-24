# Clinical ML Classification Pipeline: Breast Cancer Malignancy

## Overview
This repository contains a clinical machine learning pipeline designed to classify breast cancer malignancy. The project demonstrates an end-to-end data science workflow, from data preprocessing and dimensionality reduction to exhaustive hyperparameter tuning across multiple models.

## Key Performance Metrics
* **Test Accuracy:** 98.25% on unseen data
* **Precision:** 100% 

## Technical Architecture & Methodology
* **Feature Engineering:** Evaluated complex feature spaces using Principal Component Analysis (PCA), polynomial expansion, and SelectKBest. Applied robust scaling and L2 regularization to manage feature multicollinearity.
* **Model Selection:** Executed exhaustive grid searches to optimize bias-variance tradeoffs across three distinct architectures:
  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Multi-Layer Perceptron (MLP) Neural Networks
* **Tech Stack:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib

## Repository Structure
* `the_breast_cancer_ml_project.ipynb`: The core research notebook containing exploratory data analysis, model training, and evaluation metrics.
* `requirements.txt`: Project dependencies.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook to view the training pipeline and evaluation charts.
