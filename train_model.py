import fastf1
import pandas as pd
import numpy as np
from model import F1LapTimePredictor
import os
from sklearn.metrics import mean_absolute_error

os.makedirs('models', exist_ok=True)

# Cache setup
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# ==============================================================================
# 1. DATA LOADING (with RoundNumber for temporal ordering)
# ==============================================================================
YEARS = [2023, 2024]
SESSION = 'R'
all_races_dfs = []

print("🏁 Loading data...")
for YEAR in YEARS:
    try:
        schedule = fastf1.get_event_schedule(YEAR)
        races = schedule[~schedule['EventFormat'].str.contains('testing', case=False, na=False)]
        
        for _, race_info in races.iterrows():
            race_name = race_info['EventName']
            round_number = race_info['RoundNumber']
            
            try:
                session = fastf1.get_session(YEAR, race_name, SESSION)
                session.load(laps=True, telemetry=False, weather=True)
                df = session.laps.copy()
                
                # Merge weather
                if hasattr(session, 'weather_data') and not session.weather_data.empty:
                    df = pd.merge_asof(
                        df.sort_values('Time'), 
                        session.weather_data[['Time', 'TrackTemp']], 
                        on='Time', 
                        direction='nearest'
                    )
                
                # Add metadata for temporal ordering
                df['EventName'] = race_name
                df['RoundNumber'] = round_number + (YEAR - 2023) * 100  # Unique ordering across years
                df['Year'] = YEAR
                
                all_races_dfs.append(df)
                print(f"   ✅ Loaded {race_name} {YEAR} (Round {round_number})")
            except Exception as e:
                print(f"   ⚠️ Skipped {race_name}: {e}")
    except Exception as e:
        print(f"Failed to get schedule for {YEAR}: {e}")

if not all_races_dfs:
    print("❌ No data loaded. Exiting.")
    exit(1)

df_combined = pd.concat(all_races_dfs, ignore_index=True)
print(f"\n📊 Loaded {len(df_combined):,} laps from {len(all_races_dfs)} races")

# Ensure LapTimeSeconds exists
df_combined['LapTimeSeconds'] = df_combined['LapTime'].dt.total_seconds()

# ==============================================================================
# 2. PHYSICS LEARNING & FEATURE ENGINEERING
# ==============================================================================
predictor = F1LapTimePredictor()

print("\n" + "="*60)
print("STEP 1: LEARNING PHYSICS CONSTANTS")
predictor.fit_physics_parameters(df_combined)

print("\nSTEP 2: ENGINEERING FEATURES")
df_engineered = predictor.engineer_features(df_combined)

# ==============================================================================
# 3. TARGET NORMALIZATION (Ratio Strategy)
# ==============================================================================
print("\n" + "="*60)
print("STEP 3: CALCULATING TARGET RATIOS")

# Calculate Reference Time (fastest lap per race)
df_engineered['ReferenceTime'] = df_engineered.groupby('EventName')['LapTimeSeconds'].transform('min')
df_engineered['TargetRatio'] = df_engineered['LapTimeSeconds'] / df_engineered['ReferenceTime']

print(f"   Mean Reference Time: {df_engineered['ReferenceTime'].mean():.2f}s")
print(f"   Mean Target Ratio:   {df_engineered['TargetRatio'].mean():.4f}")

# ==============================================================================
# 4. PROPER TEMPORAL SPLIT (Race-based, not lap-based!)
# ==============================================================================
print("\n" + "="*60)
print("STEP 4: TEMPORAL RACE-BASED SPLIT")

# Use the new proper splitting method
train_df, val_df, test_df = predictor.create_train_val_test_split(
    df_engineered, 
    val_size=0.15, 
    test_size=0.15
)

# ==============================================================================
# 5. PREPARE DATA MATRICES
# ==============================================================================
print("\n" + "="*60)
print("STEP 5: PREPARING DATA MATRICES")

X_train, y_train = predictor.prepare_data(train_df, target_col='TargetRatio')
X_val, y_val = predictor.prepare_data(val_df, target_col='TargetRatio')
X_test, y_test = predictor.prepare_data(test_df, target_col='TargetRatio')

# Set feature names from training data (required for align_features)
predictor.feature_names = X_train.columns.tolist()

# Align val and test to have same features as train
X_val = predictor.align_features(X_val)
X_test = predictor.align_features(X_test)

print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# ==============================================================================
# 6. TRAIN MODEL
# ==============================================================================
print("\n" + "="*60)
print("STEP 6: TRAINING")

predictor.train(X_train, y_train, X_val, y_val, train_df=train_df, val_df=val_df)

# ==============================================================================
# 7. EVALUATE ON TRULY UNSEEN FUTURE RACES
# ==============================================================================
print("\n" + "="*60)
print("STEP 7: EVALUATION ON UNSEEN FUTURE RACES")

# Predict on full test set first
pred_ratios = predictor.model.predict(X_test)

# Add predictions to test_df for easier slicing
# We use .loc to ensure alignment
test_df_eval = test_df.loc[X_test.index].copy()
test_df_eval['PredictedRatio'] = pred_ratios
test_df_eval['PredictedTime'] = test_df_eval['PredictedRatio'] * test_df_eval['ReferenceTime']
test_df_eval['ActualTime'] = y_test * test_df_eval['ReferenceTime']

# Define Clean Lap Mask (Must match app.py logic accurately!)
# App.py filters: Lap > 1, TrackStatus=1, IsInLap=0, IsOutLap=0
# Note: IsInLap/IsOutLap were dropped from X_test but exist in test_df_eval
clean_mask = (
    (test_df_eval['TrackStatus'] == '1') & 
    (test_df_eval['LapNumber'] > 1) &
    (test_df_eval['IsInLap'] == 0) &
    (test_df_eval['IsOutLap'] == 0)
)

print(f"   📊 Evaluation Filter: Clean Laps only (TrackStatus=1, No In/Out/Pit)")

# Overall Test Metric
if clean_mask.sum() > 0:
    mae = mean_absolute_error(
        test_df_eval.loc[clean_mask, 'ActualTime'], 
        test_df_eval.loc[clean_mask, 'PredictedTime']
    )
    print(f"   🎯 GLOBAL TEST MAE: {mae:.3f}s (n={clean_mask.sum()} laps)")
else:
    print("   ⚠️ No clean laps found in test set?")

print("\n   🏁 PER-RACE BREAKDOWN:")
# Breakdown by Race
test_races = test_df_eval['EventName'].unique()
for race in test_races:
    race_mask = (test_df_eval['EventName'] == race) & clean_mask
    
    if race_mask.sum() > 0:
        race_mae = mean_absolute_error(
            test_df_eval.loc[race_mask, 'ActualTime'], 
            test_df_eval.loc[race_mask, 'PredictedTime']
        )
        print(f"      - {race:<25}: {race_mae:.3f}s (n={race_mask.sum()})")
    else:
        print(f"      - {race:<25}: N/A (No clean laps)")

print("\n" + "="*60)

# ==============================================================================
# 8. FINAL TRAINING ON ALL DATA & SAVE
# ==============================================================================
print("\n" + "="*60)
print("STEP 8: FINAL TRAINING ON ALL DATA")

# Combine all data for final model
all_df = pd.concat([train_df, val_df, test_df])
X_all, y_all = predictor.prepare_data(all_df, target_col='TargetRatio')

# Retrain on everything
predictor.train(X_all, y_all, X_all, y_all, train_df=all_df, val_df=all_df)

# Save
predictor.save_model('models/f1_model_ratio.pkl')
print("✅ Model Saved to models/f1_model_ratio.pkl")
print("\n🏎️ Run `streamlit run app.py` to test!")