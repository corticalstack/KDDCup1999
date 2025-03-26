# 🔍 KDD Cup 1999 Network Intrusion Detection System

A comprehensive machine learning framework for network intrusion detection using the KDD Cup 1999 dataset.

## 📋 Description

This repository contains a complete machine learning pipeline for analyzing network traffic and detecting intrusions using the KDD Cup 1999 dataset. The framework implements various preprocessing techniques, feature selection methods, and machine learning models to classify network connections as normal or malicious (with further categorization of attack types).

The KDD Cup 1999 dataset is a widely used benchmark for intrusion detection systems, containing a wide variety of intrusions simulated in a military network environment.

## 🚀 Features

- **Data Preprocessing**
  - Data cleaning and transformation
  - Feature engineering
  - Handling of imbalanced classes
  - Outlier detection and removal
  - Correlation analysis

- **Feature Selection**
  - Univariate selection with Chi-squared
  - Recursive feature elimination
  - Principal Component Analysis (PCA)
  - Tree-based feature importance (Random Forest, Extra Trees)

- **Machine Learning Models**
  - Neural Networks (Single Layer Perceptron, Multi-Layer Perceptron)
  - XGBoost
  - Logistic Regression
  - Clustering

- **Visualization**
  - Confusion matrices
  - Correlation heatmaps
  - Feature distribution plots
  - Decision boundaries
  - Cluster visualizations
  - 2D and 3D PCA plots

## 🔧 Prerequisites

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- XGBoost
- Matplotlib
- Seaborn
- Missingno

## 🛠️ Setup Guide

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install numpy pandas scikit-learn tensorflow xgboost matplotlib seaborn missingno
   ```
3. Download the KDD Cup 1999 dataset (a 10% subset is included in the `data` directory)
4. Run the preprocessing script to prepare the data:
   ```
   python preprocessing.py
   ```

## 📊 Usage

The framework is organized as a pipeline of Python scripts that can be run sequentially:

1. **Data Preprocessing**:
   ```
   python preprocessing.py
   ```
   This script loads the raw dataset, cleans it, transforms categorical features, and performs initial analysis.

2. **Feature Selection**:
   ```
   python featureselection.py
   ```
   This script applies various feature selection techniques and evaluates their performance.

3. **Scaling**:
   ```
   python scaling.py
   ```
   This script applies different scaling methods to the features and evaluates their impact.

4. **Sampling**:
   ```
   python sampling.py
   ```
   This script applies different sampling techniques to handle class imbalance.

5. **Model Training and Evaluation**:
   ```
   python xgboostBinary.py  # For binary classification with XGBoost
   python annMLPBinary.py   # For binary classification with neural networks
   python annMLPMultiClass.py  # For multi-class classification with neural networks
   ```

## 📁 Project Structure

- `dataset.py`: Defines the dataset class and methods for data manipulation
- `preprocessing.py`: Handles initial data processing and exploration
- `featureselection.py`: Implements various feature selection techniques
- `scaling.py`: Applies different scaling methods to the features
- `sampling.py`: Implements techniques for handling class imbalance
- `modelling.py`: Contains base model classes and evaluation methods
- `visualize.py`: Provides visualization functions for data and results
- `filehandler.py`: Handles file I/O operations
- Model implementations:
  - `xgboostBinary.py`: XGBoost for binary classification
  - `annSLPBinary.py`: Single Layer Perceptron for binary classification
  - `annMLPBinary.py`: Multi-Layer Perceptron for binary classification
  - `annMLPMultiClass.py`: Multi-Layer Perceptron for multi-class classification
  - `clustering.py`: Clustering algorithms for unsupervised learning

## 📈 Results

The repository includes extensive visualizations of model performance in the `viz` directory:
- Confusion matrices for different models and configurations
- Decision boundaries for various classifiers
- Feature importance plots
- Cluster visualizations
- PCA projections

## 📚 Resources

- [KDD Cup 1999 Dataset Information](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [Intrusion Detection Evaluation Dataset (CICIDS2017)](https://www.unb.ca/cic/datasets/ids-2017.html)
- [A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection](https://ieeexplore.ieee.org/document/7307098)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
