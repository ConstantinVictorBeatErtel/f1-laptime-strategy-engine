# F1 Strategy Command Center - Setup Guide

This guide will help you recreate the entire F1 Strategy Engine experiment from scratch.

## Prerequisites

- Python 3.9 or higher
- Git
- 8GB+ RAM recommended
- Internet connection (for downloading F1 data)

## Quick Start (Standalone - No Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/ConstantinVictorBeatErtel/f1-strategy-engine.git
cd f1-strategy-engine
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Model

This will download F1 data and train the XGBoost model:

```bash
python train_model.py
```

**What this does:**
- Downloads race datasets from FastF1 API (configured for 2023-2024 by default)
- Engineers 50+ physics-based features
- Trains XGBoost model with temporal validation
- Saves model to `models/f1_model_ratio.pkl`
- Logs experiment to MLflow

**Note:** First run will download and cache F1 data (may take time depending on connection speed). Subsequent runs use the cache.

### 5. Start MLflow (Optional)

In a separate terminal:

```bash
mlflow ui --port 5001
```

Access at: http://localhost:5001

### 6. Run the Dashboard

```bash
streamlit run app.py
```

Access at: http://localhost:8501

---

## Docker Setup (Full Microservices)

### 1. Prerequisites

- Docker Desktop installed
- Docker Compose installed

### 2. Start All Services

```bash
docker-compose up -d --build
```

### 3. Access Services

- **Streamlit Dashboard:** http://localhost:8501
- **MLflow UI:** http://localhost:5001
- **Jupyter Lab:** http://localhost:8888 (password: `f1racing`)

### 4. Stop Services

```bash
docker-compose down
```

---

## Training Data

### Data Configuration

The `train_model.py` script is configured to use:
- **Default Years:** 2023-2024 (can be modified)
- **Races:** All Grand Prix races (excludes testing)
- **Features:** 50+ engineered variables

### Features Engineered

1. **Tire Physics:**
   - Dynamic grip coefficient (temperature-adjusted)
   - Degradation rate (track surface-adjusted)
   - Compound properties (optimal temp, sensitivity)
   - Tire age effects (quadratic wear)

2. **Fuel Modeling:**
   - Fuel mass calculations
   - Corner penalty (slow corners more affected)
   - Straight-line penalty
   - Downforce × fuel interaction

3. **Track Characteristics:**
   - Average speed
   - Corner count
   - Downforce level (1-5)
   - Surface abrasiveness (1-5)
   - Elevation changes

4. **Race Context:**
   - Lap number
   - Stint progress
   - Track evolution (rubber buildup)
   - Weather conditions

### Data Sources

All data is fetched automatically from the FastF1 API:
- Lap times
- Tire compounds and age
- Track temperatures
- Pit stops
- Race positions

**No manual data download required!**

---

## File Structure

```
f1-strategy-engine/
├── app.py                    # Streamlit dashboard
├── model.py                  # ML model & feature engineering
├── train_model.py            # Training script
├── logger.py                 # Logging utility
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── SETUP.md                  # This file
├── APP_UPDATES.md           # Changelog
├── .gitignore               # Git exclusions
├── Dockerfile.app           # Docker config for Streamlit
├── Dockerfile.mlflow        # Docker config for MLflow
├── docker-compose.yml       # Orchestration
├── models/                  # Saved models (created by training)
├── cache/                   # FastF1 data cache (created automatically)
└── logs/                    # Application logs (created automatically)
```

---

## Model Training Details

### Training Pipeline

```
Load F1 Data (FastF1 API)
    ↓
Feature Engineering (50+ variables)
    ↓
Temporal Train/Val/Test Split
    ↓
XGBoost Training with Early Stopping
    ↓
MLflow Experiment Logging
    ↓
Model Persistence
```

---

## Using the Dashboard

### Tab 1: Race Pace Analysis
- Load a race (supports 2023-2025 data)
- Select a driver
- View predicted vs actual lap times
- Analyze prediction error distribution
- See tire compound usage over the race

### Tab 2: Strategy Simulator
- Set current lap position
- Choose tire compounds (current → target)
- Toggle safety car scenario
- Adjust temperature forecast
- Get optimal pit stop lap recommendations
- View pit window analysis

### Tab 3: Tire Physics
- Explore theoretical degradation curves
- Adjust temperature and track abrasiveness
- Compare compounds (SOFT/MEDIUM/HARD)
- View real race tire performance data

---

## Troubleshooting

### Issue: Model file not found

**Solution:** Run `python train_model.py` first to create the model.

### Issue: FastF1 data download is slow

**Solution:**
- This is normal - FastF1 API can be slow
- Data is cached locally after first download
- Be patient during first run

### Issue: MLflow connection error

**Solution:**
- Start MLflow server: `mlflow ui --port 5001`
- Or comment out MLflow-related code if not needed

### Issue: Streamlit import errors

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Out of memory during training

**Solution:**
- Close other applications
- Modify `YEARS` in `train_model.py` to train on fewer seasons

---

## Customization

### Train on Different Years

Edit `train_model.py`:
```python
YEARS = [2022, 2023, 2024, 2025]  # Modify as needed
```

### Add New Track Characteristics

Edit `model.py` in `TRACK_CHARACTERISTICS` dictionary:
```python
'New Circuit Name': {
    'avg_speed_kmh': 220.0,
    'corner_count': 16,
    'slow_corners_ratio': 0.25,
    'longest_straight_km': 1.0,
    'elevation_change_m': 20.0,
    'track_length_km': 5.0,
    'downforce_level': 3,
    'surface_abrasiveness': 3,
}
```

### Adjust Tire Physics Parameters

Edit `model.py` in `COMPOUND_PHYSICS` dictionary:
```python
'SOFT': {
    'grip_coefficient': 1.15,
    'base_deg_rate': 0.025,
    'optimal_temp_C': 28,
    'temp_sensitivity': 0.002,
    'warm_up_laps': 1,
    'peak_grip_lap': 2,
}
```

---

## What You'll Need to Recreate

From the GitHub repository alone, someone can:

✅ **Clone the code** - All source files are included
✅ **Install dependencies** - `requirements.txt` provided
✅ **Train the model** - `train_model.py` handles everything
✅ **Run the dashboard** - `app.py` ready to go
✅ **Use Docker** - Full Docker setup included

⚠️ **They will need to:**
- Download F1 data via FastF1 (automatic, but requires internet)
- Train the model (automatic via `train_model.py`)
- Have sufficient RAM for training

❌ **Not included in repo:**
- Pre-trained models (too large for git)
- Historical data CSVs (generated from FastF1)
- MLflow artifacts (generated during training)

---

## Performance Tips

### Faster Setup
- Use cached data (automatic after first run)
- Start with fewer years for testing
- Skip MLflow if not needed

### Better Workflow
- Train model first before running dashboard
- Use MLflow to track experiments
- Monitor logs for debugging

---

## Contributing

Found a bug or have an improvement?
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

This project is for educational purposes using open data from the FastF1 library.

---

## Support

- **GitHub Issues:** https://github.com/ConstantinVictorBeatErtel/f1-strategy-engine/issues
- **FastF1 Docs:** https://docs.fastf1.dev/
- **Streamlit Docs:** https://docs.streamlit.io/

---

**🏎️ Ready to race!**
