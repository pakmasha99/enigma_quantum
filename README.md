# Quantum Machine Learning for OCD Classification (ENIGMA-OCD)

## Project Overview

This project explores the application of **Quantum Machine Learning (QML)** to psychiatric neuroimaging, specifically focusing on the classification of **Obsessive-Compulsive Disorder (OCD)** patients versus Healthy Controls using resting-state functional MRI (rs-fMRI) time-series data.

The dataset is derived from the **ENIGMA-OCD** working group, representing the largest available OCD cohort with data from over 20-30 international study sites.

### Motivation
Prior studies using classical machine learning and deep learning on this heterogeneous, multi-site dataset have yielded modest performance (AUC ~0.55-0.67). This project investigates whether **Quantum Time-Series Transformers (QTST)** can leverage quantum entanglement and interference to capture complex temporal dependencies in fMRI signals more effectively than classical counterparts, potentially improving classification outcomes and generalizability across scanners.

## Methodology

We harness a hybrid quantum-classical architecture:
*   **QTSTransformer**: A custom Transformer model where the classical self-attention mechanism is augmented or improved by Variational Quantum Circuits (VQC) or orthogonal quantum layers to process time-series tokens.
*   **Double Stratification**: To address significant batch effects inherent in multi-site neuroimaging, our training pipeline implements a robust stratification strategy that simultaneously balances **Diagnosis** (OCD/HC) and **Acquisition Site** across Train, Validation, and Test sets.

## Repository Structure

This repository is organized into the following directories:

### `code/`
Contains the core implementation of the quantum hybrid models and training pipelines.

*   **`QTSTransformer.py`**: Defines the model architecture classes, including the Quantum Time-Series Transformer layers and quantum circuit definitions.
*   **`QTSTransformer_train.py`**: The main training engine. Handles data loading, site-aware stratification, training loops (training, validation, testing), and metric logging (AUC, Sensitivity, Specificity).
*   **`QTSTransformer_tuning.py`**: A hyperparameter tuning script (using Optuna framework) to optimize quantum circuit depth, learning rates, and transformer architecture parameters.

### `papers/`
A collection of essential literature and references relevant to this project, including:
*   Foundational papers on Quantum Machine Learning in neuroimaging.
*   Key studies from the ENIGMA-OCD working group.
*   Technical references for Quantum Transformers.
