#  F1 Race Strategy Predictor

An End-to-End Machine Learning project that predicts Formula 1 race pace based on tire degradation and fuel burn.

##  Architecture
This project runs as a microservices architecture using Docker containers:
* **Database:** PostgreSQL (Stores lap time data)
* **Model:** XGBoost (Regression model trained on Bahrain 2024 data)
* **Tracking:** MLflow (Experiment tracking and model registry)
* **Frontend:** Streamlit (Interactive Strategy Dashboard)
* **Infrastructure:** Docker Compose (Orchestration)

##  How to Run

### Step 1: Clone the repository
You need to download the code to your machine.
```bash
git clone [https://github.com/YOUR_USERNAME/f1-strategy-predictor.git](https://github.com/YOUR_USERNAME/f1-strategy-predictor.git)
cd f1-strategy-predictor
```

### Step 2: Start the App
This command builds the Docker containers and launches the services.
```bash
docker-compose up -d --build
```
*Note: The first build may take a few minutes to download dependencies.*

### Step 3: Access the Services
Once the containers are running, you can access them in your browser:
* **Dashboard:** http://localhost:8501
* **Jupyter Lab:** http://localhost:8888 (Password: `f1racing`)
* **MLflow UI:** http://localhost:5001

## 📊 Workflow
1. **Ingestion:** Data is fetched from the FastF1 API and stored in Postgres (`01_data_ingestion.ipynb`).
2. **Training:** An XGBoost model is trained on `TyreLife`, `LapNumber` (Fuel effect), `Compound`, and `Team` (`02_model_training.ipynb`).
3. **Deployment:** The best model is logged to MLflow and loaded dynamically by the Streamlit Dashboard.

## 🛠️ Tech Stack
* **Python 3.9**
* **Docker & Docker Compose**
* **PostgreSQL**
* **MLflow**
* **Streamlit**
* **XGBoost**
* **Pandas / SQLAlchemy**

## 📝 License
This project is for educational purposes using open data from the FastF1 library.
