# F1 Race Strategy Predictor

An End-to-End Machine Learning project that predicts Formula 1 race pace based on tire degradation and fuel burn.

## Architecture
* **Database:** PostgreSQL (Stores lap time data)
* **Model:** XGBoost (Regression model trained on Bahrain 2024 data)
* **Tracking:** MLflow (Experiment tracking and model registry)
* **Frontend:** Streamlit (Interactive Strategy Dashboard)
* **Infrastructure:** Docker Compose (Orchestration)

## How to Run
Prerequisite: Install [Docker Desktop](https://www.docker.com/products/docker-desktop).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/f1-strategy-predictor.git](https://github.com/YOUR_USERNAME/f1-strategy-predictor.git)
   cd f1-strategy-predictor
