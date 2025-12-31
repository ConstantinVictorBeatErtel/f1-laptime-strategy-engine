import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import os
from logger import setup_logger

# Create logger instance
logger = setup_logger("f1_model")

# ==============================================================================
# TRACK CHARACTERISTICS DATABASE
# ==============================================================================

TRACK_CHARACTERISTICS = {
    'Albert Park Circuit': {
        'avg_speed_kmh': 235.0,
        'corner_count': 14,
        'slow_corners_ratio': 0.21,
        'longest_straight_km': 1.3,
        'elevation_change_m': 2.6,
        'track_length_km': 5.278,
        'downforce_level': 3,
        'surface_abrasiveness': 2,
    },
    'Shanghai International Circuit': {
        'avg_speed_kmh': 205.0,
        'corner_count': 16,
        'slow_corners_ratio': 0.37,
        'longest_straight_km': 1.17,
        'elevation_change_m': 11.0,
        'track_length_km': 5.451,
        'downforce_level': 3,
        'surface_abrasiveness': 3,
    },
    'Suzuka Circuit': {
        'avg_speed_kmh': 225.0,
        'corner_count': 18,
        'slow_corners_ratio': 0.11,
        'longest_straight_km': 0.8,
        'elevation_change_m': 40.4,
        'track_length_km': 5.807,
        'downforce_level': 4,
        'surface_abrasiveness': 4,
    },
    'Bahrain International Circuit': {
        'avg_speed_kmh': 210.0,
        'corner_count': 15,
        'slow_corners_ratio': 0.33,
        'longest_straight_km': 1.09,
        'elevation_change_m': 18.0,
        'track_length_km': 5.412,
        'downforce_level': 3,
        'surface_abrasiveness': 5,
    },
    'Jeddah Corniche Circuit': {
        'avg_speed_kmh': 252.0,
        'corner_count': 27,
        'slow_corners_ratio': 0.07,
        'longest_straight_km': 0.8,
        'elevation_change_m': 2.0,
        'track_length_km': 6.174,
        'downforce_level': 2,
        'surface_abrasiveness': 2,
    },
    'Miami International Autodrome': {
        'avg_speed_kmh': 220.0,
        'corner_count': 19,
        'slow_corners_ratio': 0.31,
        'longest_straight_km': 1.28,
        'elevation_change_m': 4.0,
        'track_length_km': 5.412,
        'downforce_level': 3,
        'surface_abrasiveness': 2,
    },
    'Imola Circuit': {
        'avg_speed_kmh': 225.0,
        'corner_count': 19,
        'slow_corners_ratio': 0.21,
        'longest_straight_km': 0.6,
        'elevation_change_m': 33.3,
        'track_length_km': 4.909,
        'downforce_level': 4,
        'surface_abrasiveness': 3,
    },
    'Circuit de Monaco': {
        'avg_speed_kmh': 160.0,
        'corner_count': 19,
        'slow_corners_ratio': 0.52,
        'longest_straight_km': 0.5,
        'elevation_change_m': 42.0,
        'track_length_km': 3.337,
        'downforce_level': 5,
        'surface_abrasiveness': 1,
    },
    'Circuit de Barcelona-Catalunya': {
        'avg_speed_kmh': 215.0,
        'corner_count': 14,
        'slow_corners_ratio': 0.21,
        'longest_straight_km': 1.05,
        'elevation_change_m': 29.6,
        'track_length_km': 4.657,
        'downforce_level': 4,
        'surface_abrasiveness': 4,
    },
    'Circuit Gilles Villeneuve': {
        'avg_speed_kmh': 210.0,
        'corner_count': 14,
        'slow_corners_ratio': 0.35,
        'longest_straight_km': 1.06,
        'elevation_change_m': 5.2,
        'track_length_km': 4.361,
        'downforce_level': 2,
        'surface_abrasiveness': 2,
    },
    'Red Bull Ring': {
        'avg_speed_kmh': 235.0,
        'corner_count': 10,
        'slow_corners_ratio': 0.30,
        'longest_straight_km': 0.8,
        'elevation_change_m': 63.5,
        'track_length_km': 4.318,
        'downforce_level': 3,
        'surface_abrasiveness': 3,
    },
    'Silverstone Circuit': {
        'avg_speed_kmh': 240.0,
        'corner_count': 18,
        'slow_corners_ratio': 0.11,
        'longest_straight_km': 0.77,
        'elevation_change_m': 11.3,
        'track_length_km': 5.891,
        'downforce_level': 4,
        'surface_abrasiveness': 5,
    },
    'Spa-Francorchamps': {
        'avg_speed_kmh': 245.0,
        'corner_count': 19,
        'slow_corners_ratio': 0.15,
        'longest_straight_km': 2.0,
        'elevation_change_m': 102.2,
        'track_length_km': 7.004,
        'downforce_level': 2,
        'surface_abrasiveness': 4,
    },
    'Hungaroring': {
        'avg_speed_kmh': 195.0,
        'corner_count': 14,
        'slow_corners_ratio': 0.42,
        'longest_straight_km': 0.9,
        'elevation_change_m': 34.7,
        'track_length_km': 4.381,
        'downforce_level': 5,
        'surface_abrasiveness': 3,
    },
    'Zandvoort': {
        'avg_speed_kmh': 215.0,
        'corner_count': 14,
        'slow_corners_ratio': 0.28,
        'longest_straight_km': 0.68,
        'elevation_change_m': 15.0,
        'track_length_km': 4.259,
        'downforce_level': 5,
        'surface_abrasiveness': 4,
    },
    'Monza': {
        'avg_speed_kmh': 260.0,
        'corner_count': 11,
        'slow_corners_ratio': 0.18,
        'longest_straight_km': 1.12,
        'elevation_change_m': 12.8,
        'track_length_km': 5.793,
        'downforce_level': 1,
        'surface_abrasiveness': 2,
    },
    'Baku City Circuit': {
        'avg_speed_kmh': 210.0,
        'corner_count': 20,
        'slow_corners_ratio': 0.45,
        'longest_straight_km': 2.2,
        'elevation_change_m': 26.8,
        'track_length_km': 6.003,
        'downforce_level': 2,
        'surface_abrasiveness': 1,
    },
    'Marina Bay Street Circuit': {
        'avg_speed_kmh': 190.0,
        'corner_count': 19,
        'slow_corners_ratio': 0.58,
        'longest_straight_km': 0.8,
        'elevation_change_m': 5.3,
        'track_length_km': 4.940,
        'downforce_level': 5,
        'surface_abrasiveness': 3,
    },
    'Circuit of the Americas': {
        'avg_speed_kmh': 205.0,
        'corner_count': 20,
        'slow_corners_ratio': 0.35,
        'longest_straight_km': 1.0,
        'elevation_change_m': 41.0,
        'track_length_km': 5.513,
        'downforce_level': 4,
        'surface_abrasiveness': 3,
    },
    'Autodromo Hermanos Rodriguez': {
        'avg_speed_kmh': 195.0,
        'corner_count': 17,
        'slow_corners_ratio': 0.47,
        'longest_straight_km': 1.2,
        'elevation_change_m': 2.8,
        'track_length_km': 4.304,
        'downforce_level': 5,
        'surface_abrasiveness': 2,
    },
    'Interlagos': {
        'avg_speed_kmh': 215.0,
        'corner_count': 15,
        'slow_corners_ratio': 0.26,
        'longest_straight_km': 0.65,
        'elevation_change_m': 43.0,
        'track_length_km': 4.309,
        'downforce_level': 4,
        'surface_abrasiveness': 4,
    },
    'Las Vegas Strip Circuit': {
        'avg_speed_kmh': 238.0,
        'corner_count': 17,
        'slow_corners_ratio': 0.23,
        'longest_straight_km': 1.9,
        'elevation_change_m': 2.0,
        'track_length_km': 6.201,
        'downforce_level': 1,
        'surface_abrasiveness': 2,
    },
    'Lusail International Circuit': {
        'avg_speed_kmh': 220.0,
        'corner_count': 16,
        'slow_corners_ratio': 0.12,
        'longest_straight_km': 1.06,
        'elevation_change_m': 5.0,
        'track_length_km': 5.419,
        'downforce_level': 4,
        'surface_abrasiveness': 4,
    },
    'Yas Marina Circuit': {
        'avg_speed_kmh': 200.0,
        'corner_count': 16,
        'slow_corners_ratio': 0.31,
        'longest_straight_km': 1.2,
        'elevation_change_m': 10.7,
        'track_length_km': 5.281,
        'downforce_level': 3,
        'surface_abrasiveness': 3,
    }
}

# ==============================================================================
# COMPOUND PHYSICS DATABASE (Updated with temperature sensitivity)
# ==============================================================================
COMPOUND_PHYSICS = {
    'SOFT': {
        'grip_coefficient': 1.15,           # Relative grip multiplier
        'base_deg_rate': 0.025,            # Base degradation per lap
        'optimal_temp_C': 28,              # Optimal track temp
        'temp_sensitivity': 0.002,         # Grip loss per °C away from optimal
        'warm_up_laps': 1,
        'peak_grip_lap': 2,
    },
    'MEDIUM': {
        'grip_coefficient': 1.00,
        'base_deg_rate': 0.015,
        'optimal_temp_C': 35,
        'temp_sensitivity': 0.0015,
        'warm_up_laps': 2,
        'peak_grip_lap': 3,
    },
    'HARD': {
        'grip_coefficient': 0.88,
        'base_deg_rate': 0.010,
        'optimal_temp_C': 42,
        'temp_sensitivity': 0.001,
        'warm_up_laps': 3,
        'peak_grip_lap': 4,
    },
    'INTERMEDIATE': {
        'grip_coefficient': 0.75,
        'base_deg_rate': 0.008,
        'optimal_temp_C': 18,
        'temp_sensitivity': 0.003,
        'warm_up_laps': 1,
        'peak_grip_lap': 1,
    },
    'WET': {
        'grip_coefficient': 0.60,
        'base_deg_rate': 0.005,
        'optimal_temp_C': 15,
        'temp_sensitivity': 0.004,
        'warm_up_laps': 1,
        'peak_grip_lap': 1,
    },
}


def calculate_dynamic_tire_performance(compound, track_temp, track_abrasiveness, tyre_life):
    """
    FIX 1: Calculate tire performance dynamically based on conditions
    This replaces static compound feature lookups
    
    Returns:
        current_grip: Adjusted grip coefficient accounting for temperature and age
        adjusted_deg_rate: Degradation rate scaled by track abrasiveness
    """
    if compound not in COMPOUND_PHYSICS:
        # Default to MEDIUM if compound unknown
        compound = 'MEDIUM'
    
    props = COMPOUND_PHYSICS[compound]
    
    # Temperature penalty (quadratic - worse further from optimal)
    temp_delta = abs(track_temp - props['optimal_temp_C'])
    temp_penalty = (temp_delta ** 1.5) * props['temp_sensitivity']
    
    # Degradation scaled by track surface
    # Abrasive tracks (5) wear tires 2x faster than smooth tracks (1)
    abrasiveness_multiplier = track_abrasiveness / 3.0  # Normalized to 3 (medium)
    adjusted_deg_rate = props['base_deg_rate'] * abrasiveness_multiplier
    
    # Calculate current grip accounting for tire age
    # Grip peaks at peak_grip_lap, then degrades
    base_grip = props['grip_coefficient']
    
    if tyre_life <= props['peak_grip_lap']:
        # Warming up phase - linear improvement
        age_factor = min(1.0, tyre_life / props['peak_grip_lap'])
    else:
        # Degradation phase - exponential decay (wear accelerates)
        laps_past_peak = tyre_life - props['peak_grip_lap']
        # Quadratic degradation: wear accelerates over time
        age_factor = 1.0 - (adjusted_deg_rate * laps_past_peak * (1 + 0.02 * laps_past_peak))
        age_factor = max(0.3, age_factor)  # Minimum 30% grip remaining
    
    current_grip = base_grip * age_factor - temp_penalty
    current_grip = max(0.2, current_grip)  # Floor at 20%
    
    return current_grip, adjusted_deg_rate


class F1LapTimePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.deg_rates = {k: v['base_deg_rate'] for k, v in COMPOUND_PHYSICS.items()}
        self.driver_baselines = {}
    
    def fit_physics_parameters(self, df):
        """
        Load physics constants and calculate driver baselines
        """
        print("🔬 Loading Physics Constants & Driver Baselines...")
        
        # Degradation rates already loaded in __init__
        print(f"   ✓ Tire Degradation Rates: {self.deg_rates}")
        
        # Calculate driver skill baselines (5th percentile of clean laps)
        clean_laps = df[
            (df['TrackStatus'] == '1') & 
            (df.get('IsInLap', 0) == 0) & 
            (df.get('IsOutLap', 0) == 0) &
            (df['LapNumber'] > 1)
        ]
        
        if 'LapTimeSeconds' in clean_laps.columns and len(clean_laps) > 0:
            self.driver_baselines = clean_laps.groupby('Driver')['LapTimeSeconds'].quantile(0.05).to_dict()
            print(f"   ✓ Calculated baselines for {len(self.driver_baselines)} drivers")
        else:
            print("   ⚠️ Could not calculate driver baselines")
    
    def engineer_features(self, df):
        """
        Enhanced feature engineering with dynamic physics calculations
        """
        df = df.copy()
        print("🛠️  Engineering features with dynamic physics...")
        
        # ======================================================================
        # 1. BASIC CONVERSIONS
        # ======================================================================
        if 'LapTime' in df.columns and not pd.api.types.is_numeric_dtype(df['LapTime']):
            df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        
        # ======================================================================
        # 2. TRACK STATUS
        # ======================================================================
        if 'TrackStatus' in df.columns:
            df['TrackStatus'] = df['TrackStatus'].astype(str)
            df['TrackStatus_Impact'] = df['TrackStatus'].map({
                '1': 0,   # Green
                '2': 1,   # Yellow
                '6': 2, '7': 2,  # VSC
                '4': 3, '5': 3   # SC
            }).fillna(0)
        
        # ======================================================================
        # 3. TRACK CHARACTERISTICS
        # ======================================================================
        if 'EventName' in df.columns:
            for feature in ['avg_speed_kmh', 'corner_count', 'slow_corners_ratio', 
                           'longest_straight_km', 'elevation_change_m', 'track_length_km',
                           'downforce_level', 'surface_abrasiveness']:
                df[f'Track_{feature}'] = df['EventName'].map(
                    {k: v.get(feature, np.nan) for k, v in TRACK_CHARACTERISTICS.items()}
                )
            
            missing_tracks = df[df['Track_avg_speed_kmh'].isna()]['EventName'].unique()
            if len(missing_tracks) > 0:
                print(f"   ⚠️ Missing track data for: {missing_tracks}")
                for col in df.columns:
                    if col.startswith('Track_'):
                        df[col].fillna(df[col].median(), inplace=True)
        
        # ======================================================================
        # 4. FIX 1: DYNAMIC TIRE PERFORMANCE (Replaces static compound features)
        # ======================================================================
        print("   🔧 Calculating dynamic tire physics...")
        
        # Ensure we have required columns with defaults
        if 'TrackTemp' not in df.columns:
            df['TrackTemp'] = 30.0  # Default moderate temp
        if 'Track_surface_abrasiveness' not in df.columns:
            df['Track_surface_abrasiveness'] = 3.0  # Default medium
        if 'Compound' not in df.columns:
            df['Compound'] = 'MEDIUM'
        
        # Calculate dynamic tire performance for each lap
        tire_calcs = df.apply(
            lambda row: calculate_dynamic_tire_performance(
                row['Compound'],
                row['TrackTemp'],
                row['Track_surface_abrasiveness'],
                row.get('TyreLife', 1)
            ),
            axis=1
        )
        
        df['CurrentTireGrip'] = [x[0] for x in tire_calcs]
        df['DynamicDegRate'] = [x[1] for x in tire_calcs]
        
        # Add static compound properties for reference (but as separate features)
        for feature in ['optimal_temp_C', 'temp_sensitivity', 'warm_up_laps', 'peak_grip_lap']:
            df[f'Compound_{feature}'] = df['Compound'].map(
                {k: v[feature] for k, v in COMPOUND_PHYSICS.items()}
            ).fillna(COMPOUND_PHYSICS['MEDIUM'][feature])
        
        # ======================================================================
        # 5. FUEL LOAD MODELING
        # ======================================================================
        if 'EventName' in df.columns:
            race_max_laps = df.groupby('EventName')['LapNumber'].transform('max')
        else:
            race_max_laps = df['LapNumber'].max()
        
        df['RaceProgress'] = df['LapNumber'] / race_max_laps
        df['FuelLoad_Proxy'] = 1.1 - df['RaceProgress']
        df['FuelMass_kg'] = df['FuelLoad_Proxy'] * 110  # Assume 110kg max fuel
        
        # ======================================================================
        # 6. PIT LAP FLAGS
        # ======================================================================
        if 'PitOutTime' in df.columns:
            df['IsOutLap'] = (~df['PitOutTime'].isna()).astype(int)
        else:
            df['IsOutLap'] = 0
        
        if 'PitInTime' in df.columns:
            df['IsInLap'] = (~df['PitInTime'].isna()).astype(int)
        else:
            df['IsInLap'] = 0
        
        df['FreshTyres'] = (df.get('TyreLife', 1) == 1).astype(int)
        
        # ======================================================================
        # 7. FIX 2: ADVANCED TIRE DEGRADATION FEATURES
        # ======================================================================
        print("   🔧 Engineering degradation interaction features...")
        
        # Grip remaining (already calculated dynamically)
        df['GripRemaining'] = df['CurrentTireGrip']
        
        # Degradation acceleration (quadratic - wear accelerates)
        df['DegAcceleration'] = (df.get('TyreLife', 1) ** 2) * df['Track_surface_abrasiveness'] * 0.001
        
        # Temperature × Degradation interaction
        df['TempDegInteraction'] = (
            df['TrackTemp'] * df.get('TyreLife', 1) * df['DynamicDegRate']
        )
        
        # Expected tire loss (legacy feature, kept for compatibility)
        df['Expected_Tyre_Loss'] = df.get('TyreLife', 1) * df['DynamicDegRate']
        
        # Tire grip curve (peak, then degrade)
        df['TireGrip_Curve'] = df['CurrentTireGrip'] / df['Compound'].map(
            {k: v['grip_coefficient'] for k, v in COMPOUND_PHYSICS.items()}
        ).fillna(1.0)
        
        # ======================================================================
        # 8. FIX 3: COMPREHENSIVE TRACK-TIRE-FUEL INTERACTIONS
        # ======================================================================
        print("   🔧 Engineering physics interaction features...")
        
        # TIRE-TRACK INTERACTIONS
        # How well does the tire match the track surface?
        df['TireTrackMatch'] = df['CurrentTireGrip'] * (
            1 / (df['Track_surface_abrasiveness'] + 1)
        )
        
        # Adjusted degradation accounting for track
        df['DegPerLap_Adjusted'] = df['DynamicDegRate'] * df['Track_surface_abrasiveness']
        
        # FUEL-TRACK INTERACTIONS
        # Fuel penalty varies by track characteristics
        df['FuelCornerPenalty'] = (
            df['FuelMass_kg'] * df['Track_slow_corners_ratio'] * 0.0003
        )
        
        df['FuelStraightPenalty'] = (
            df['FuelMass_kg'] * (df['Track_longest_straight_km'] / 2.0) * 0.001
        )
        
        df['FuelLapEffect'] = df['FuelCornerPenalty'] + df['FuelStraightPenalty']
        
        # Fuel squared (acceleration penalty is non-linear)
        df['FuelLoad_Squared'] = df['FuelLoad_Proxy'] ** 2
        
        # DOWNFORCE-FUEL TRADE-OFF
        # High downforce tracks are more sensitive to fuel weight
        df['FuelDownforceLoad'] = (
            df['FuelLoad_Proxy'] * (df['Track_downforce_level'] / 5.0)
        )
        
        # Corner-specific fuel tax (slow corners = more fuel penalty)
        df['CornerFuelTax'] = df['FuelMass_kg'] * df['Track_slow_corners_ratio']
        
        # TEMPERATURE EFFECTS
        # Temperature mismatch severity (scaled by track surface)
        df['TempMismatch_Severity'] = (
            abs(df['TrackTemp'] - df['Compound_optimal_temp_C']) * 
            df['Track_surface_abrasiveness'] * 0.01
        )
        
        # Continuous temperature matching score
        df['Temp_Compound_Match'] = df.apply(lambda row:
            max(0, 1 - abs(row['TrackTemp'] - row['Compound_optimal_temp_C']) * 
                row['Compound_temp_sensitivity']),
            axis=1
        )
        
        # COMPOUND-TRACK WEAR INTERACTION
        df['Track_Compound_Wear'] = (
            df['Track_surface_abrasiveness'] * df['DynamicDegRate']
        )
        
        # FUEL × TIRE WEAR (Heavy car stresses tires more)
        df['Fuel_TireWear_Interaction'] = df['FuelLoad_Proxy'] * df['Expected_Tyre_Loss']
        
        # SPEED × TIRE STRESS (Fast tracks harder on tires)
        df['Speed_TireStress'] = df['Track_avg_speed_kmh'] * df['DynamicDegRate'] * 0.001
        
        # ======================================================================
        # 9. TRACK EVOLUTION
        # ======================================================================
        if 'SessionType' in df.columns:
            df['SessionLap'] = df.groupby(['EventName', 'SessionType']).cumcount() + 1
            df['TrackEvolution_Effect'] = -0.15 * np.log(df['SessionLap'] + 1)
            df.loc[df['SessionType'] == 'Race', 'TrackEvolution_Effect'] *= 0.3
        else:
            df['SessionLap'] = df.groupby('EventName').cumcount() + 1
            df['TrackEvolution_Effect'] = -0.05 * np.log(df['SessionLap'] + 1)
        
        # ======================================================================
        # 10. WEATHER CONDITIONS
        # ======================================================================
        if 'Rainfall' in df.columns:
            df['IsWet'] = (df['Rainfall'] > 0).astype(int)
        
        # ======================================================================
        # 11. DRIVER SKILL NORMALIZATION
        # ======================================================================
        if 'Driver' in df.columns and len(self.driver_baselines) > 0:
            df['Driver_Baseline'] = df['Driver'].map(self.driver_baselines)
            median_baseline = np.median(list(self.driver_baselines.values()))
            df['Driver_Baseline'].fillna(median_baseline, inplace=True)
        
        # ======================================================================
        # 12. STINT CONTEXT
        # ======================================================================
        if all(col in df.columns for col in ['EventName', 'Driver', 'LapNumber']):
            df = df.sort_values(['EventName', 'Driver', 'LapNumber'])
            df['StintChange'] = (df.get('TyreLife', 1) == 1).astype(int)
            df['StintNumber'] = df.groupby(['EventName', 'Driver'])['StintChange'].cumsum()
            df['LapsSincePit (Tire age proxy)'] = df.groupby(['EventName', 'Driver', 'StintNumber']).cumcount()
            
            # Expected stint length based on compound
            df['ExpectedStintLength'] = df['Compound'].map({
                k: v['peak_grip_lap'] * 6 for k, v in COMPOUND_PHYSICS.items()
            }).fillna(20)
            
            df['StintProgress'] = df['LapsSincePit (Tire age proxy)'] / df['ExpectedStintLength']
        
        # ======================================================================
        # 13. ADDITIONAL COMPOUND FEATURES (For warm-up modeling)
        # ======================================================================
        if 'TyreLife' in df.columns:
            df['Compound_warm_up_laps'] = df['Compound'].map({
                k: v['warm_up_laps'] for k, v in COMPOUND_PHYSICS.items()
            }).fillna(2)
            
            # Binary: Is tire still warming up?
            df['IsWarmingUp'] = (df['TyreLife'] <= df['Compound_warm_up_laps']).astype(int)
            
            # Warm-up progress (0 to 1)
            df['WarmUpProgress'] = (df['TyreLife'] / df['Compound_warm_up_laps']).clip(upper=1.0)
        
        print(f"   ✓ Engineered {len(df.columns)} total features")
        logger.info(f"Feature engineering complete: {len(df.columns)} total features created from {len(df)} laps")
        print(f"   ✓ Engineered {len(df.columns)} total features")
        return df
        return df
    
    def prepare_data(self, df, target_col='LapTimeSeconds', drop_cols=None):
        """
        Clean data preparation WITHOUT track/driver identity features
        """
        print("🧹 Preparing data matrix...")
        
        # Drop bad targets
        if target_col in df.columns:
            original_len = len(df)
            df = df.dropna(subset=[target_col])
            df = df[np.isfinite(df[target_col])]
            
            if target_col == 'TargetRatio':
                df = df[df[target_col] < 1.5]
            
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                print(f"   ⚠️ Dropped {dropped_rows} rows with invalid {target_col}")
                logger.warning(f"Dropped {dropped_rows} rows with invalid {target_col} values")
        
        # ==================================================================
        # ONLY ENCODE TEAM
        # ==================================================================
        if 'Team' in df.columns:
            team_dummies = pd.get_dummies(df['Team'], prefix='Team', drop_first=False)
            df = pd.concat([df, team_dummies], axis=1)
        
        # ==================================================================
        # LEAKAGE COLUMNS
        # ==================================================================
        LEAKAGE_COLUMNS = [
            # Targets
            'LapTime', 'LapTimeSeconds', 'Time', 'TargetRatio',
            # Sector times
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            # Speed measurements
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
            # Session timing
            'LapStartTime', 'LapStartDate',
            # Pit times
            'PitOutTime', 'PitInTime',
            # Identity features
            'Driver', 'EventName', 'Compound',
            # Metadata
            'SessionType', 'Year', 'RoundNumber', 'SessionLap',
            'StintChange', 'StintNumber',
            # Filter columns
            'IsPersonalBest', 'Deleted', 'DeletedReason',
            'IsAccurate', 'TrackStatus', 'TrackStatus_Impact',
            # Original flags
            'IsInLap', 'IsOutLap', 'FreshTyres',
            'Compound_optimal_temp_C',
            'Compound_temp_sensitivity', 
            'Compound_warm_up_laps',
            'Compound_peak_grip_lap', 
            'ExpectedStintLength'
        ]
        
        # Drop leakage
        drop_cols_final = [col for col in LEAKAGE_COLUMNS if col in df.columns]
        
        if 'ReferenceTime' in df.columns and target_col == 'TargetRatio':
            drop_cols_final.append('ReferenceTime')
        
        X = df.drop(columns=drop_cols_final + [target_col], errors='ignore')
        y = df[target_col]
        
        # Drop non-numeric
        non_numeric = X.select_dtypes(include=['object', 'datetime']).columns
        if len(non_numeric) > 0:
            print(f"   ⚠️ Dropping non-numeric columns: {non_numeric.tolist()}")
            X = X.drop(columns=non_numeric)
        
        # Fill NaN
        if X.isna().sum().sum() > 0:
            print(f"   ⚠️ Filling {X.isna().sum().sum()} NaN values with 0")
            X = X.fillna(0)
        
        print(f"   ✓ Feature matrix: {X.shape}")
        
        return X, y
    
    def align_features(self, X_new):
        """Ensure test/val sets have same features as training"""
        if self.feature_names is None:
            raise ValueError("Must set feature_names from training data first")
        
        # Add missing columns with 0
        for col in self.feature_names:
            if col not in X_new.columns:
                X_new[col] = 0
        
        # Remove extra columns
        X_new = X_new[self.feature_names]
        
        return X_new
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        logger.info(f"Training started with {X_train.shape[0]:,} samples and {X_train.shape[1]} features")
        print(f"🎯 Training XGBoost on {X_train.shape[0]:,} samples...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='mae'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Final metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"   ✓ Train MAE: {train_mae:.4f}")
        print(f"   ✓ Val MAE:   {val_mae:.4f}")
        logger.info(f"Training complete - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

        
        return self.model
    
    def temporal_train_test_split(self, df, test_size=0.2):
        """Split data temporally by race"""
        print(f"⏰ Temporal split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
        
        df_temp = df.copy()
        df_temp['RaceID'] = df_temp['Year'].astype(str) + "_" + df_temp['EventName']
        
        if 'RoundNumber' in df_temp.columns:
            race_order = df_temp.groupby('RaceID')['RoundNumber'].first().sort_values()
            ordered_race_ids = race_order.index.tolist()
        else:
            ordered_race_ids = sorted(df_temp['RaceID'].unique())
        
        n_races = len(ordered_race_ids)
        split_idx = int(n_races * (1 - test_size))
        
        train_ids = ordered_race_ids[:split_idx]
        test_ids = ordered_race_ids[split_idx:]
        
        train_df = df_temp[df_temp['RaceID'].isin(train_ids)].drop(columns=['RaceID'])
        test_df = df_temp[df_temp['RaceID'].isin(test_ids)].drop(columns=['RaceID'])
        
        print(f"   ✓ Train: {len(train_ids)} races ({len(train_df):,} laps)")
        print(f"   ✓ Test:  {len(test_ids)} races ({len(test_df):,} laps)")
        
        return train_df, test_df
    
    def create_train_val_test_split(self, df, val_size=0.15, test_size=0.15):
        """Create train/val/test splits"""
        temp_df, test_df = self.temporal_train_test_split(df, test_size=test_size)
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = self.temporal_train_test_split(temp_df, test_size=val_size_adjusted)
        
        print(f"\n📈 Final split:")
        print(f"   Train: {len(train_df):,} laps")
        print(f"   Val:   {len(val_df):,} laps")
        print(f"   Test:  {len(test_df):,} laps")
        
        return train_df, val_df, test_df
    
    def save_model(self, filepath):
        """Save model and metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'deg_rates': self.deg_rates,
                'driver_baselines': self.driver_baselines,
            }, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and metadata"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.deg_rates = data.get('deg_rates', self.deg_rates)
            self.driver_baselines = data.get('driver_baselines', {})
        print(f"✅ Model loaded from {filepath}")