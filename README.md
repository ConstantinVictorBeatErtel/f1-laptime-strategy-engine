# 🏎️ F1 Strategy Command Center

An advanced Machine Learning project that predicts Formula 1 race pace and optimizes pit stop strategy using physics-informed AI.

## ✨ Key Features

- **Dynamic Tire Physics:** Real-time grip degradation based on compound, temperature, and track abrasiveness
- **Strategy Optimization:** AI-powered pit stop timing with lap-by-lap simulation
- **Race Pace Prediction:** XGBoost model with 50+ engineered features
- **Interactive Dashboard:** Beautiful Formula One-themed Streamlit interface
- **MLflow Integration:** Complete experiment tracking and model versioning

## 🏗️ Architecture
This project runs as a microservices architecture using Docker containers:
* **Database:** PostgreSQL (Stores lap time data)
* **Model:** XGBoost Regressor with Physics-Based Features
* **Tracking:** MLflow (Experiment tracking and model registry)
* **Frontend:** Streamlit (Premium F1-Styled Dashboard)
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

## 📊 Model Features

The XGBoost model uses physics-based feature engineering:

### Physics Features
- **Dynamic Tire Performance:** Temperature-adjusted grip coefficients
- **Degradation Modeling:** Quadratic wear with track abrasiveness
- **Fuel Effects:** Non-linear fuel load impact on lap times
- **Track Characteristics:** 24 circuits with detailed profiles
- **Temperature Sensitivity:** Compound-specific optimal temperatures

### Feature Categories (50+ total)
- Tire dynamics (grip, degradation, compound properties)
- Fuel load modeling (mass, cornering penalty, straight-line penalty)
- Track characteristics (speed, corners, downforce, surface)
- Weather conditions (temperature, rainfall)
- Race context (lap number, stint progress, track evolution)

## 🎨 Dashboard Features

### Tab 1: Race Pace Analysis
- Real-time lap time predictions vs actual
- Multi-panel visualization (lap times + tire age)
- Error distribution analysis
- 4-metric performance dashboard

### Tab 2: Strategy Simulator
- Interactive pit stop optimization
- Safety car scenario modeling
- Temperature-adjusted predictions
- Visual pit window recommendations

### Tab 3: Tire Physics
- Theoretical degradation curves
- Temperature sensitivity analysis
- Real race tire performance tracking
- Compound comparison tables

## 🛠️ Tech Stack
* **Python 3.9+**
* **Docker & Docker Compose**
* **PostgreSQL** (Data storage)
* **MLflow** (Experiment tracking)
* **Streamlit** (Interactive dashboard)
* **XGBoost** (ML model)
* **FastF1** (F1 data API)
* **Plotly** (Advanced visualizations)
* **Pandas / NumPy** (Data processing)

## 📝 License
This project is for educational purposes using open data from the FastF1 library.
