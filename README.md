# 🏠 Housing Price MLOps: Production-Grade Regression Pipeline

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=flat-square&logo=mlflow)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/Infrastructure-AWS-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://www.python.org/)

An end-to-end Machine Learning system designed to predict housing prices using **XGBoost**. This project transitions from a standalone model to a robust **MLOps architecture**, featuring automated hyperparameter tuning, containerized microservices, and a cloud-native deployment strategy on AWS.



---

## 🏗️ System Architecture

The codebase is organized into decoupled, modular pipelines to ensure scalability and ease of maintenance.

### 🔄 The MLOps Lifecycle
`Load` → `Preprocess` → `Feature Engineering` → `Tune` → `Register` → `Serve` → `Monitor`

| Component | Responsibility | Tech Stack |
| :--- | :--- | :--- |
| **Feature Pipeline** | Time-aware splitting, city normalization, & target encoding. | `Pandas`, `Scikit-learn` |
| **Training Pipeline** | Bayesian optimization & experiment tracking. | `XGBoost`, `Optuna`, `MLflow` |
| **Inference Layer** | High-performance REST API & Interactive UI. | `FastAPI`, `Streamlit`, `S3` |
| **Infrastructure** | Container orchestration & CI/CD. | `Docker`, `AWS ECS (Fargate)`, `GitHub Actions` |

---

## 🛠️ Core Engineering Features

### 1. Robust Data Integrity & Leakage Prevention
To ensure model reliability in real estate forecasting, the system enforces:
* **Temporal Splits:** Strict time-based indexing (Train: <2020 | Eval: 2020-21 | Holdout: 2022+).
* **Encoder Persistence:** Frequency and Target encoders are fitted *only* on training data and serialized as artifacts for consistent inference transformations.
* **Schema Alignment:** Automated validation between training and real-time inference data structures.

### 2. Experiment Tracking & Optimization
The training pipeline is fully automated to find the global minimum of the loss function:
* **Automated Tuning:** Uses **Optuna** to search for optimal XGBoost hyperparameters ($learning\_rate$, $max\_depth$, etc.).
* **MLflow Registry:** All metrics (MAE, RMSE, % Error), artifacts, and model versions are stored in an S3-backed MLflow server.

### 3. Cloud-Native Deployment (AWS)
* **Elastic Container Service (ECS):** Serverless execution of the API and Dashboard using Fargate.
* **Application Load Balancer (ALB):** Orchestrates traffic between the **FastAPI** backend (port 8000) and **Streamlit** frontend (port 8501).
* **S3-First Strategy:** All model weights and datasets are synced dynamically from `housing-regression-data` buckets.

---

## 🚀 Execution Guide

### 1. Environment Setup
Managed by **`uv`** for lightning-fast, reproducible dependency resolution.
```bash
uv sync