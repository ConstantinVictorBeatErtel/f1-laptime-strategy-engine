import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import F1LapTimePredictor, TRACK_CHARACTERISTICS, COMPOUND_PHYSICS
import os
from logger import setup_logger

# Create logger
logger = setup_logger("f1_predictions")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="F1 Strategy Command Center",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS (PREMIUM F1 THEME) ---
st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #15151d 100%);
        color: #FFFFFF;
    }

    /* Headers with Racing Font Style */
    h1 {
        color: #E10600;
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
        font-weight: 900;
        font-size: 3.5rem !important;
        letter-spacing: -2px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #E10600 0%, #ff4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 2rem !important;
        border-left: 5px solid #E10600;
        padding-left: 15px;
        margin-top: 2rem;
    }

    h3 {
        color: #cccccc;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
        font-size: 1.3rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #E10600 0%, #c00500 100%);
        color: white;
        border-radius: 8px;
        border: 2px solid #ff3333;
        font-weight: 900;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #ff3333 0%, #E10600 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(225, 6, 0, 0.6);
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #E10600;
        font-size: 2.5rem !important;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(225, 6, 0, 0.5);
    }

    div[data-testid="stMetricLabel"] {
        color: #999999;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b1b22 0%, #0f0f14 100%);
        border-right: 2px solid #E10600;
    }

    section[data-testid="stSidebar"] h2 {
        border-left: none;
        padding-left: 0;
        color: #E10600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(27, 27, 34, 0.5);
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #999999;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E10600 0%, #c00500 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(225, 6, 0, 0.4);
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(27, 27, 34, 0.8);
        border: 1px solid #333;
        border-left: 4px solid #E10600;
        border-radius: 6px;
        padding: 1rem;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #E10600 0%, #ff4444 100%);
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: rgba(27, 27, 34, 0.8);
        border: 1px solid #333;
        border-radius: 6px;
    }

    /* Racing stripe decoration */
    .race-stripe {
        height: 4px;
        background: linear-gradient(90deg,
            transparent 0%,
            #E10600 10%,
            #E10600 90%,
            transparent 100%);
        margin: 2rem 0;
        box-shadow: 0 0 10px rgba(225, 6, 0, 0.5);
    }

    /* Card styling */
    .card {
        background: rgba(27, 27, 34, 0.6);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE & SETUP ---
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

@st.cache_resource
def load_model():
    predictor = F1LapTimePredictor()
    # Ensure this matches your saved model path
    if os.path.exists('models/f1_model_ratio.pkl'):
        predictor.load_model('models/f1_model_ratio.pkl')
        return predictor
    return None

@st.cache_data
def load_race_data(year, gp, session_type='R'):
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=False, weather=True)
        return session
    except Exception as e:
        return None

# --- MAIN APP LOGIC ---
def main():
    # Header with racing stripe
    st.title("🏎️ F1 Strategy Command Center")
    st.markdown("### Physics-Informed AI for Race Strategy Optimization")
    st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)

    # Sidebar: Controls
    with st.sidebar:
        st.image("https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/F1%20logo%20red.png", width=200)
        st.markdown("---")
        st.header("🎛️ Race Configuration")

        # Year selection
        year = st.selectbox("🏁 Season", [2025, 2024, 2023], index=0)
        
        # Dynamic Schedule Loading
        try:
            schedule = fastf1.get_event_schedule(year)
            # Filter out testing
            races = schedule[~schedule['EventFormat'].str.contains('testing', case=False, na=False)]
            
            # Create display names with Train/Test indicator
            # Hybrid Split Logic: First 5 races are updated into training, rest are test
            UPDATE_AFTER_N_RACES = 5
            
            race_options = {}
            for _, r in races.iterrows():
                round_num = r['RoundNumber']
                name = r['EventName']
                
                if round_num <= UPDATE_AFTER_N_RACES:
                    status = "📚 TRAIN (Update Phase)"
                else:
                    status = "🧪 TEST (Unseen Future)"
                    
                label = f"{name} (Round {round_num}) [{status}]"
                race_options[label] = name
            
            selected_label = st.selectbox("Grand Prix", list(race_options.keys()), index=0)
            gp = race_options[selected_label]
            
        except Exception as e:
            st.error(f"Could not load schedule: {e}")
            gp = 'Bahrain Grand Prix' # Fallback
        
        load_btn = st.button("🚀 Load Race Data")

        st.markdown("---")

        # Display track characteristics if GP selected
        if 'gp' in locals() and gp in TRACK_CHARACTERISTICS:
            with st.expander("📍 Track Info"):
                track_info = TRACK_CHARACTERISTICS[gp]
                st.metric("Track Length", f"{track_info['track_length_km']:.2f} km")
                st.metric("Average Speed", f"{track_info['avg_speed_kmh']:.0f} km/h")
                st.metric("Corners", track_info['corner_count'])
                st.metric("Downforce Level", f"{track_info['downforce_level']}/5")

        st.markdown("---")
        st.info("""
        **🧠 Model Architecture:**
        - **Type:** XGBoost Regressor
        - **Physics:** Dynamic Tire Degradation
        - **Features:** 50+ engineered variables
        - **Training:** Temporal Split
        - **Tracking:** MLflow Integration
        """)

        with st.expander("🔧 Tire Compound Physics"):
            st.markdown("**Degradation Rates (s/lap):**")
            for compound, props in COMPOUND_PHYSICS.items():
                st.text(f"{compound:12s}: {props['base_deg_rate']:.3f} | Optimal: {props['optimal_temp_C']}°C")


    # Load Model
    predictor = load_model()
    if not predictor:
        st.error("❌ Model file `models/f1_model_ratio.pkl` not found. Please train the model first.")
        return

    # Session State management for data
    if 'session' not in st.session_state:
        st.session_state.session = None

    if load_btn:
        with st.spinner(f"📥 Loading {gp} {year}..."):
            st.session_state.session = load_race_data(year, gp)

    if st.session_state.session:
        session = st.session_state.session
        df = session.laps.copy()
        
        # --- CRITICAL DATA PREP (Fixing the errors we saw earlier) ---
        df['EventName'] = session.event['EventName']
        # round_number might need casting
        round_num = session.event['RoundNumber']
        df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        
        # Display Train/Test Status
        # Hybrid Split Logic: First 5 races are updated into training, rest are test
        if round_num > 5:
             st.warning(f"🧪 **TEST SET (UNSEEN DATA):** This 2025 race (Round {round_num}) is in the future relative to training.")
        else:
             st.success(f"📚 **TRAINING SET (UPDATE PHASE):** The model has been updated with results from this early 2025 race (Round {round_num}).")
        
        # Merge Weather
        if hasattr(session, 'weather_data') and not session.weather_data.empty:
            df = pd.merge_asof(df.sort_values('Time'), session.weather_data[['Time', 'TrackTemp']], on='Time', direction='nearest')

        # Driver Selection
        drivers = df['Driver'].unique()
        selected_driver = st.sidebar.selectbox("Select Driver", drivers, index=0)
        
        # Engineer Features
        df_eng = predictor.engineer_features(df)
        
        # Calculate Reference Time (for ratio conversion)
        ref_time = df_eng['LapTimeSeconds'].min()
        df_eng['ReferenceTime'] = ref_time
        
        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Race Pace Analysis", "🧠 Strategy Simulator", "🔬 Tire Physics"])

        # ==============================================================================
        # TAB 1: RACE PACE VALIDATION (Enhanced with multi-driver comparison)
        # ==============================================================================
        with tab1:
            st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)
            st.subheader(f"🎯 Race Pace Validation: {selected_driver}")

            # Filter for Driver
            d_df = df_eng[df_eng['Driver'] == selected_driver].copy()

            # Predict
            X_input, _ = predictor.prepare_data(d_df)
            X_aligned = predictor.align_features(X_input)
            pred_ratios = predictor.model.predict(X_aligned)
            logger.info(f"Race pace prediction for {selected_driver} at {gp}: {len(pred_ratios)} laps predicted")

            # Convert Ratio -> Seconds
            d_df['PredictedTime'] = np.nan
            d_df.loc[X_input.index, 'PredictedTime'] = pred_ratios * d_df.loc[X_input.index, 'ReferenceTime']

            # --- CLEAN PACE FILTER ---
            clean_mask = (
                (d_df['LapNumber'] > 1) &
                (d_df['TrackStatus'] == '1') &
                (d_df['IsInLap'] == 0) &
                (d_df['IsOutLap'] == 0) &
                (d_df['PredictedTime'].notna())
            )

            clean_df = d_df[clean_mask]

            # Enhanced Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            if not clean_df.empty:
                mae = np.mean(np.abs(clean_df['LapTimeSeconds'] - clean_df['PredictedTime']))
                rmse = np.sqrt(np.mean((clean_df['LapTimeSeconds'] - clean_df['PredictedTime'])**2))
                max_error = np.max(np.abs(clean_df['LapTimeSeconds'] - clean_df['PredictedTime']))

                col1.metric("MAE (Clean Air)", f"{mae:.3f}s", help="Mean Absolute Error on clean laps")
                col2.metric("RMSE", f"{rmse:.3f}s", help="Root Mean Squared Error")
                col3.metric("Max Error", f"{max_error:.3f}s", help="Largest single lap prediction error")
                col4.metric("Clean Laps", len(clean_df), help="Laps used for validation")
            else:
                st.warning("⚠️ Not enough clean laps to calculate metrics.")

            # Enhanced Plotting with tire stint visualization
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=("Lap Time Prediction", "Tire Age & Compound"),
                vertical_spacing=0.12
            )

            # Top plot: Lap times
            fig.add_trace(go.Scatter(
                x=clean_df['LapNumber'],
                y=clean_df['LapTimeSeconds'],
                mode='markers',
                name='Actual Time',
                marker=dict(color='rgba(150, 150, 150, 0.6)', size=8, line=dict(width=1, color='white')),
                hovertemplate='Lap %{x}<br>Actual: %{y:.3f}s<extra></extra>'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=clean_df['LapNumber'],
                y=clean_df['PredictedTime'],
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='#E10600', width=3),
                marker=dict(size=4, color='#E10600'),
                hovertemplate='Lap %{x}<br>Predicted: %{y:.3f}s<extra></extra>'
            ), row=1, col=1)

            # Bottom plot: Tire compound visualization
            compound_colors = {
                'SOFT': '#FF0000',
                'MEDIUM': '#FFFF00',
                'HARD': '#FFFFFF',
                'INTERMEDIATE': '#00FF00',
                'WET': '#0000FF'
            }

            for compound in clean_df['Compound'].unique():
                compound_laps = clean_df[clean_df['Compound'] == compound]
                fig.add_trace(go.Scatter(
                    x=compound_laps['LapNumber'],
                    y=compound_laps['TyreLife'],
                    mode='markers',
                    name=compound,
                    marker=dict(
                        size=10,
                        color=compound_colors.get(compound, '#888888'),
                        symbol='square'
                    ),
                    hovertemplate=f'{compound}<br>Lap: %{{x}}<br>Tire Age: %{{y}} laps<extra></extra>',
                    showlegend=True
                ), row=2, col=1)

            fig.update_xaxes(title_text="Lap Number", row=2, col=1)
            fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Tire Age", row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=700,
                showlegend=True,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,30,0.5)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Error distribution histogram
            if not clean_df.empty:
                st.markdown("### 📈 Prediction Error Distribution")
                errors = clean_df['LapTimeSeconds'] - clean_df['PredictedTime']

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=30,
                    marker=dict(
                        color='#E10600',
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Error: %{x:.3f}s<br>Count: %{y}<extra></extra>'
                ))

                fig_hist.update_layout(
                    title="Distribution of Prediction Errors",
                    xaxis_title="Error (s) - Negative = Predicted Faster",
                    yaxis_title="Frequency",
                    template="plotly_dark",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(20,20,30,0.5)'
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # ==============================================================================
        # TAB 2: STRATEGY SIMULATOR (Enhanced UI)
        # ==============================================================================
        with tab2:
            st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)
            st.subheader("⚡ Strategy Optimizer")
            
            # Enhanced input section with cards
            st.markdown("### ⚙️ Simulation Parameters")

            col1, col2, col3 = st.columns(3)
            with col1:
                max_lap = int(df['LapNumber'].max() - 5)
                current_lap = st.slider("📍 Current Lap", min_value=5, max_value=max_lap, value=20)
                st.caption(f"Race length: {int(df['LapNumber'].max())} laps")

            with col2:
                is_sc = st.toggle("🚨 Safety Car Deployed?", value=False)
                pit_loss = 11.0 if is_sc else 20.0
                st.metric("Pit Stop Time Loss", f"{pit_loss}s", help="Time lost in pit lane")

            with col3:
                current_temp = df['TrackTemp'].iloc[-1] if 'TrackTemp' in df.columns else 35.0
                future_temp = st.slider("🌡️ Track Temp Forecast (°C)", min_value=15, max_value=55, value=int(current_temp))
                st.caption("Account for weather changes")

            st.markdown("### 🏁 Tire Strategy")
            c1, c2, c3 = st.columns(3)
            with c1:
                current_compound = st.selectbox("🔴 Current Compound", ['SOFT', 'MEDIUM', 'HARD'], index=1,
                    help="Tire compound currently on the car")
            with c2:
                target_compound = st.selectbox("🎯 Target Compound", ['SOFT', 'MEDIUM', 'HARD'], index=2,
                    help="Compound to switch to at pit stop")
            with c3:
                # Show current tire age
                if selected_driver in df_eng['Driver'].values:
                    current_tire_age = df_eng.loc[
                        (df_eng['Driver'] == selected_driver) &
                        (df_eng['LapNumber'] == current_lap), 'TyreLife'
                    ].values
                    if len(current_tire_age) > 0:
                        st.metric("Current Tire Age", f"{int(current_tire_age[0])} laps")
                    else:
                        st.metric("Current Tire Age", "Unknown")

            # Strategy summary card
            st.info(f"🎯 **Simulation:** {selected_driver} | {current_compound} → {target_compound} | Track Temp: {future_temp}°C | {'🚨 SC Active' if is_sc else '🟢 Green Flag'}")

            if st.button("🚀 Run Strategy Simulation"):
                
                # 1. CALIBRATE TO SPECIFIC DRIVER
                # We look at the laps the driver has ALREADY driven to find their "Pace Factor"
                # This makes the simulation respect that Norris is faster than Sargeant
                driver_history = df_eng[
                    (df_eng['Driver'] == selected_driver) & 
                    (df_eng['LapNumber'] <= current_lap) &
                    (df_eng['TrackStatus'] == '1') &  # Clean laps only
                    (df_eng['IsInLap'] == 0) & 
                    (df_eng['IsOutLap'] == 0)
                ]
                
                # If we have history, calculate how they perform vs the Reference
                if not driver_history.empty:
                    # Calculate actual average ratio
                    actual_ratio = (driver_history['LapTimeSeconds'] / driver_history['ReferenceTime']).mean()
                    # Calculate what the model *thinks* they should do
                    X_hist, _ = predictor.prepare_data(driver_history)
                    X_hist_aligned = predictor.align_features(X_hist)
                    pred_hist_ratio = predictor.model.predict(X_hist_aligned).mean()
                    
                    # Calibration Factor: If driver is driving 1% faster than model predicts, keep that bonus!
                    pace_calibration = actual_ratio / pred_hist_ratio
                    st.success(f"Driver Calibration: {selected_driver} is driving at **{pace_calibration:.3f}x** relative to model predictions.")
                else:
                    pace_calibration = 1.0
                    st.warning("Not enough clean history to calibrate driver pace. Using generic model.")

                # --- SIMULATION LOGIC ---
                total_laps = int(df['LapNumber'].max())
                possible_pit_laps = range(current_lap + 1, total_laps - 2)
                results = []
                
                # Progress bar
                prog_bar = st.progress(0)
                
                # Future Template
                future_laps = np.arange(current_lap, total_laps + 1)
                
                # Get the specific Reference Time for this Race
                # (This ensures 'Monaco' produces different times than 'Spa')
                race_ref_time = df_eng['ReferenceTime'].min()
                
                # FIX: Get the driver's Team from the actual data
                driver_team = df[df['Driver'] == selected_driver]['Team'].iloc[0]

                base_sim_df = pd.DataFrame({
                    'Driver': selected_driver,
                    'Team': driver_team,  # CRITICAL FIX: Add Team for one-hot encoding
                    'EventName': df['EventName'].iloc[0],  # FIX #2: Event for one-hot
                    'LapNumber': future_laps,
                    'TrackStatus': '1',
                    'PitInTime': None, 'PitOutTime': None,
                    'IsInLap': 0, 'IsOutLap': 0,
                    'TrackTemp': future_temp,  # FIX #3: Use forecast temp
                    'ReferenceTime': race_ref_time
                })
                
                # --- PHYSICS-BASED DEGRADATION RATES ---
                # These are industry-standard estimates for 2023-2024 F1
                # The ML model doesn't weight these heavily enough, so we add them explicitly
                DEG_RATES = {
                    'SOFT': 0.15,     # Softs lose ~0.15s per lap
                    'MEDIUM': 0.09,   # Mediums lose ~0.09s per lap  
                    'HARD': 0.045,    # Hards lose ~0.045s per lap
                }
                
                for i, pit_lap in enumerate(possible_pit_laps):
                    sim_df = base_sim_df.copy()
                    
                    # STINT 1: Current compound (user-selected) getting old
                    current_tire_age = df_eng.loc[
                        (df_eng['Driver'] == selected_driver) & 
                        (df_eng['LapNumber'] == current_lap), 'TyreLife'
                    ].values[0] if not df_eng.loc[
                        (df_eng['Driver'] == selected_driver) & 
                        (df_eng['LapNumber'] == current_lap)
                    ].empty else 10
                    
                    mask_stint1 = sim_df['LapNumber'] <= pit_lap
                    sim_df.loc[mask_stint1, 'Compound'] = current_compound
                    laps_driven_in_sim = sim_df.loc[mask_stint1, 'LapNumber'] - current_lap
                    sim_df.loc[mask_stint1, 'TyreLife'] = current_tire_age + laps_driven_in_sim
                    
                    # STINT 2: Target compound (user-selected) fresh
                    mask_stint2 = sim_df['LapNumber'] > pit_lap
                    sim_df.loc[mask_stint2, 'Compound'] = target_compound
                    sim_df.loc[mask_stint2, 'TyreLife'] = sim_df.loc[mask_stint2, 'LapNumber'] - pit_lap
                    
                    # Engineer & Predict (base pace from ML)
                    sim_eng = predictor.engineer_features(sim_df)
                    X_sim, _ = predictor.prepare_data(sim_eng)
                    X_sim_aligned = predictor.align_features(X_sim)
                    
                    # ML predicts base ratios
                    base_preds = predictor.model.predict(X_sim_aligned)
                    logger.info(f"Strategy simulation for {selected_driver}: Pit lap {pit_lap}, {current_compound}->{target_compound}, predicted avg: {base_preds.mean():.3f}")

                    
                    # --- PHYSICS ADJUSTMENT (THE FIX) ---
                    # Calculate explicit tire degradation penalty for each lap
                    # This ADDS to the ML prediction to make tire age matter
                    
                    # Get tire life for each lap in the simulation
                    tire_ages = sim_df['TyreLife'].values
                    compounds = sim_df['Compound'].values
                    
                    deg_penalties = np.array([
                        DEG_RATES.get(compound, 0.09) * age 
                        for age, compound in zip(tire_ages, compounds)
                    ])
                    
                    # Convert base ratio predictions to seconds
                    predicted_lap_times = (base_preds * pace_calibration) * race_ref_time
                    
                    # Add physics-based degradation penalty (in seconds)
                    # Note: We only add HALF of the full penalty because the ML model already captures SOME degradation
                    predicted_lap_times = predicted_lap_times + (deg_penalties * 0.5)
                    
                    total_time = predicted_lap_times.sum() + pit_loss
                    
                    results.append({'PitLap': pit_lap, 'TotalTime': total_time})
                    
                    # Update progress
                    prog_bar.progress((i + 1) / len(possible_pit_laps))
                
                prog_bar.empty()
                
                # --- ENHANCED RESULTS VISUALIZATION ---
                res_df = pd.DataFrame(results)

                # Calculate Delta to Best
                res_df['Delta'] = res_df['TotalTime'] - res_df['TotalTime'].min()
                best_strat = res_df.loc[res_df['TotalTime'].idxmin()]
                worst_strat = res_df.loc[res_df['TotalTime'].idxmax()]

                st.markdown("---")
                st.markdown("### 🏆 Optimal Strategy Results")

                # Enhanced metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("🏆 Optimal Pit Lap", f"Lap {int(best_strat['PitLap'])}")
                with c2:
                    st.metric("⏱️ Predicted Time", f"{best_strat['TotalTime']:.1f}s")
                with c3:
                    worst_delta = worst_strat['TotalTime'] - best_strat['TotalTime']
                    st.metric("⚠️ Worst Case Delta", f"+{worst_delta:.1f}s",
                        help="Time loss if pitting at worst lap")
                with c4:
                    # Calculate pit window (laps within 1s of optimal)
                    pit_window = res_df[res_df['Delta'] <= 1.0]
                    st.metric("✅ Pit Window", f"{len(pit_window)} laps",
                        help="Laps within 1s of optimal strategy")

                # Create enhanced visualization with annotations
                fig_strat = go.Figure()

                # Main strategy curve
                fig_strat.add_trace(go.Scatter(
                    x=res_df['PitLap'],
                    y=res_df['Delta'],
                    mode='lines+markers',
                    name='Time Lost vs Optimal',
                    line=dict(
                        color='#00ff00' if is_sc else '#E10600',
                        width=4,
                        shape='spline'
                    ),
                    marker=dict(size=6, symbol='circle'),
                    fill='tozeroy',
                    fillcolor='rgba(225, 6, 0, 0.1)' if not is_sc else 'rgba(0, 255, 0, 0.1)',
                    hovertemplate='<b>Pit Lap %{x}</b><br>Time Lost: +%{y:.2f}s<extra></extra>'
                ))

                # Highlight optimal pit lap
                fig_strat.add_trace(go.Scatter(
                    x=[best_strat['PitLap']],
                    y=[0],
                    mode='markers',
                    name='Optimal Pit Window',
                    marker=dict(
                        size=20,
                        color='gold',
                        symbol='star',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>OPTIMAL</b><br>Lap %{x}<extra></extra>'
                ))

                # Add pit window shading (within 1s of optimal)
                if len(pit_window) > 0:
                    fig_strat.add_vrect(
                        x0=pit_window['PitLap'].min() - 0.5,
                        x1=pit_window['PitLap'].max() + 0.5,
                        fillcolor="rgba(0, 255, 0, 0.1)",
                        layer="below",
                        line_width=0,
                        annotation_text="Safe Pit Window",
                        annotation_position="top left"
                    )

                fig_strat.update_layout(
                    title=f"🎯 Pit Strategy Optimization: {selected_driver}",
                    xaxis_title="Pit Stop Lap Number",
                    yaxis_title="Time Lost vs Optimal Strategy (s)",
                    template="plotly_dark",
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(20,20,30,0.5)',
                    hovermode='x unified',
                    annotations=[
                        dict(
                            x=best_strat['PitLap'],
                            y=0,
                            xref="x",
                            yref="y",
                            text=f"Optimal: Lap {int(best_strat['PitLap'])}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="gold",
                            ax=0,
                            ay=-40,
                            font=dict(size=12, color="gold")
                        )
                    ]
                )

                st.plotly_chart(fig_strat, use_container_width=True)

                # Strategy recommendations
                st.markdown("### 💡 Strategy Recommendations")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Optimal Strategy:**")
                    st.success(f"""
                    - Pit on **Lap {int(best_strat['PitLap'])}**
                    - Switch to **{target_compound}** compound
                    - Expected race time: **{best_strat['TotalTime']:.1f}s**
                    - Pit window: **Lap {int(pit_window['PitLap'].min())} - {int(pit_window['PitLap'].max())}**
                    """)

                with col2:
                    st.markdown("**⚠️ Avoid:**")
                    # Find worst 3 pit laps
                    worst_laps = res_df.nlargest(3, 'Delta')
                    st.warning(f"""
                    - Worst pit lap: **Lap {int(worst_strat['PitLap'])}** (+{worst_strat['Delta']:.1f}s)
                    - Avoid pitting too early or too late
                    - Stay within the safe pit window for flexibility
                    """)

                # Show top 5 strategies in a table
                with st.expander("📊 View Top 10 Alternative Strategies"):
                    top_strategies = res_df.nsmallest(10, 'TotalTime')[['PitLap', 'TotalTime', 'Delta']]
                    top_strategies['PitLap'] = top_strategies['PitLap'].astype(int)
                    top_strategies['TotalTime'] = top_strategies['TotalTime'].round(2)
                    top_strategies['Delta'] = top_strategies['Delta'].apply(lambda x: f"+{x:.2f}s" if x > 0 else f"{x:.2f}s")
                    top_strategies.columns = ['Pit Lap', 'Total Time (s)', 'Delta']
                    st.dataframe(top_strategies, use_container_width=True, hide_index=True)

        # ==============================================================================
        # TAB 3: TIRE PHYSICS ANALYSIS
        # ==============================================================================
        with tab3:
            st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)
            st.subheader("🔬 Tire Physics & Degradation Analysis")

            # Tire compound comparison
            st.markdown("### Compound Characteristics Comparison")

            # Create comparison table
            compound_data = []
            for compound, props in COMPOUND_PHYSICS.items():
                compound_data.append({
                    'Compound': compound,
                    'Grip Coefficient': props['grip_coefficient'],
                    'Degradation Rate': f"{props['base_deg_rate']:.3f}",
                    'Optimal Temp (°C)': props['optimal_temp_C'],
                    'Temp Sensitivity': f"{props['temp_sensitivity']:.4f}",
                    'Warm-up Laps': props['warm_up_laps'],
                    'Peak Grip Lap': props['peak_grip_lap']
                })

            compound_df = pd.DataFrame(compound_data)
            st.dataframe(compound_df, use_container_width=True, hide_index=True)

            # Tire degradation curve visualization
            st.markdown("### 📉 Theoretical Degradation Curves")

            col1, col2 = st.columns(2)
            with col1:
                track_temp_sim = st.slider("Track Temperature (°C)", 15, 50, 30, key="tire_temp")
            with col2:
                track_abrasive = st.slider("Track Abrasiveness", 1, 5, 3, key="tire_abr")

            # Simulate tire performance over stint
            max_laps = 50
            lap_range = np.arange(1, max_laps + 1)

            fig_deg = go.Figure()

            compound_colors = {
                'SOFT': '#FF0000',
                'MEDIUM': '#FFFF00',
                'HARD': '#FFFFFF',
                'INTERMEDIATE': '#00FF00',
                'WET': '#0000FF'
            }

            from model import calculate_dynamic_tire_performance

            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                grip_values = []
                for lap in lap_range:
                    grip, _ = calculate_dynamic_tire_performance(
                        compound, track_temp_sim, track_abrasive, lap
                    )
                    grip_values.append(grip)

                fig_deg.add_trace(go.Scatter(
                    x=lap_range,
                    y=grip_values,
                    mode='lines',
                    name=compound,
                    line=dict(color=compound_colors[compound], width=3),
                    hovertemplate=f'{compound}<br>Lap: %{{x}}<br>Grip: %{{y:.3f}}<extra></extra>'
                ))

            fig_deg.update_layout(
                title=f"Tire Grip vs Age (Track: {track_temp_sim}°C, Abrasiveness: {track_abrasive}/5)",
                xaxis_title="Tire Age (Laps)",
                yaxis_title="Grip Coefficient",
                template="plotly_dark",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,30,0.5)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_deg, use_container_width=True)

            # Temperature sensitivity chart
            st.markdown("### 🌡️ Temperature Sensitivity")

            temp_range = np.arange(15, 51, 1)
            fig_temp = go.Figure()

            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                grip_at_temps = []
                for temp in temp_range:
                    grip, _ = calculate_dynamic_tire_performance(
                        compound, temp, 3, 5  # Mid-stint on medium abrasive track
                    )
                    grip_at_temps.append(grip)

                fig_temp.add_trace(go.Scatter(
                    x=temp_range,
                    y=grip_at_temps,
                    mode='lines',
                    name=compound,
                    line=dict(color=compound_colors[compound], width=3),
                    hovertemplate=f'{compound}<br>Temp: %{{x}}°C<br>Grip: %{{y:.3f}}<extra></extra>'
                ))

            fig_temp.update_layout(
                title="Grip vs Track Temperature (5 lap old tires)",
                xaxis_title="Track Temperature (°C)",
                yaxis_title="Grip Coefficient",
                template="plotly_dark",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20,20,30,0.5)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # Real race data tire analysis
            if not d_df.empty:
                st.markdown("### 🏁 Race Tire Performance")

                # Calculate actual vs theoretical grip
                tire_analysis = d_df[clean_mask].copy()

                if not tire_analysis.empty and 'CurrentTireGrip' in tire_analysis.columns:
                    fig_real_tire = go.Figure()

                    for compound in tire_analysis['Compound'].unique():
                        comp_data = tire_analysis[tire_analysis['Compound'] == compound]

                        fig_real_tire.add_trace(go.Scatter(
                            x=comp_data['TyreLife'],
                            y=comp_data['CurrentTireGrip'],
                            mode='markers',
                            name=f'{compound} (Actual)',
                            marker=dict(
                                size=8,
                                color=compound_colors.get(compound, '#888888'),
                                opacity=0.6
                            ),
                            hovertemplate=f'{compound}<br>Age: %{{x}}<br>Grip: %{{y:.3f}}<extra></extra>'
                        ))

                    fig_real_tire.update_layout(
                        title=f"Actual Tire Grip in Race - {selected_driver}",
                        xaxis_title="Tire Age (Laps)",
                        yaxis_title="Calculated Grip Coefficient",
                        template="plotly_dark",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(20,20,30,0.5)'
                    )
                    st.plotly_chart(fig_real_tire, use_container_width=True)

        # ==============================================================================
        # FEATURE IMPORTANCE CHART (Bottom of Dashboard)
        # ==============================================================================
        st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)
        st.subheader("🔬 Model Feature Importance")
        st.caption("Shows which factors the ML model uses most to predict lap times (excluding identity features)")
        
        try:
            # Get feature importance from XGBoost
            importance_gain = predictor.model.get_booster().get_score(importance_type='gain')
            
            # Map internal feature names to actual names
            feature_map = {f'f{i}': name for i, name in enumerate(predictor.feature_names)}
            
            # Create DataFrame
            imp_df = pd.DataFrame({
                'Feature': [feature_map.get(k, k) for k in importance_gain.keys()],
                'Importance': list(importance_gain.values())
            })
            
            # Filter out identity features (these dominate but aren't insightful)
            ignored_prefixes = ['Team_']
            
            physics_df = imp_df[~imp_df['Feature'].str.startswith(tuple(ignored_prefixes))]
            
            # Sort and take top 15
            physics_df = physics_df.sort_values('Importance', ascending=True).tail(15)
            
            # Create horizontal bar chart
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Bar(
                x=physics_df['Importance'],
                y=physics_df['Feature'],
                orientation='h',
                marker=dict(color='#E10600'),
                hovertemplate='%{y}: %{x:.1f}<extra></extra>'
            ))
            
            fig_imp.update_layout(
                title="Top 15 Physics/Weather Features (Gain-based Importance)",
                xaxis_title="Predictive Power (Gain)",
                yaxis_title="Feature",
                template="plotly_dark",
                height=450,
                margin=dict(l=200)  # Extra left margin for long feature names
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Show identity features separately in expander
            with st.expander("🏷️ Identity Features (Driver/Team/Track)"):
                identity_df = imp_df[imp_df['Feature'].str.startswith(tuple(ignored_prefixes))]
                if not identity_df.empty:
                    identity_df = identity_df.sort_values('Importance', ascending=False).head(10)
                    st.dataframe(identity_df, use_container_width=True, hide_index=True)
                    st.caption("These features capture driver/team/track-specific performance differences.")
                else:
                    st.info("No identity features found in top predictors.")

        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

    # Footer
    st.markdown('<div class="race-stripe"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p style='font-size: 0.9rem;'>
            <strong>F1 Strategy Command Center</strong> | Powered by XGBoost & Physics-Based Feature Engineering
        </p>
        <p style='font-size: 0.8rem;'>
            Data: FastF1 API | ML Tracking: MLflow | Visualization: Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()