import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import F1LapTimePredictor
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="F1 Strategy Brain", page_icon="🏎️", layout="wide")

# --- CUSTOM CSS (F1 THEME) ---
st.markdown("""
<style>
    .stApp { background-color: #101015; color: #FFFFFF; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Arial', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #E10600; color: white; border-radius: 4px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #ff3333; }
    div[data-testid="stMetricValue"] { color: #E10600; font-size: 28px; }
    div[data-testid="stMetricLabel"] { color: #aaaaaa; }
    .css-1d391kg { background-color: #1b1b22; } /* Sidebar */
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
    st.title("🏎️ F1 AI Strategy Engine")
    st.markdown("Physics-Informed Pace Prediction & Strategy Optimization")

    # Sidebar: Controls
    with st.sidebar:
        st.header("Race Configuration")
        # STRICTLY 2025 as requested
        year = st.selectbox("Year", [2025], index=0)
        
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
        
        load_btn = st.button("Load Race Data")
        
        st.markdown("---")
        st.info("""
        **🧠 Model Architecture:**
        - **Type:** XGBoost Regressor (Physics-Informed)
        - **Training:** Temporal Split (Past -> Future)
        - **Features:** Tire Deg, Fuel, Track Temp, Rubbering
        - **Metric:** MAE on Clean Laps
        """)

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
        tab1, tab2 = st.tabs(["📊 Race Pace Analysis", "🧠 Strategy Simulator"])

        # ==============================================================================
        # TAB 1: RACE PACE VALIDATION (The "Clean Air" View)
        # ==============================================================================
        with tab1:
            st.subheader(f"Race Pace Validation: {selected_driver}")
            
            # Filter for Driver
            d_df = df_eng[df_eng['Driver'] == selected_driver].copy()
            
            # Predict
            X_input, _ = predictor.prepare_data(d_df)
            X_aligned = predictor.align_features(X_input) # Ensure columns match training
            pred_ratios = predictor.model.predict(X_aligned)
            
            # Convert Ratio -> Seconds
            d_df['PredictedTime'] = np.nan
            # Map back using index to handle dropped rows
            d_df.loc[X_input.index, 'PredictedTime'] = pred_ratios * d_df.loc[X_input.index, 'ReferenceTime']
            
            # --- CLEAN PACE FILTER ---
            # Remove Lap 1, Pit Laps, and Safety Car Laps for the metric/plot
            clean_mask = (
                (d_df['LapNumber'] > 1) & 
                (d_df['TrackStatus'] == '1') & 
                (d_df['IsInLap'] == 0) & 
                (d_df['IsOutLap'] == 0) & 
                (d_df['PredictedTime'].notna())
            )
            
            clean_df = d_df[clean_mask]
            
            # Metrics
            if not clean_df.empty:
                mae = np.mean(np.abs(clean_df['LapTimeSeconds'] - clean_df['PredictedTime']))
                st.markdown(f"**Clean Air MAE:** `{mae:.3f}s` (Excluding L1, Pits, SC)")
            else:
                st.warning("Not enough clean laps to calculate MAE.")

            # Plotting
            fig = go.Figure()

            # Actual Laps (Grey Dots)
            fig.add_trace(go.Scatter(
                x=clean_df['LapNumber'], 
                y=clean_df['LapTimeSeconds'],
                mode='markers',
                name='Actual Time',
                marker=dict(color='rgba(255, 255, 255, 0.5)', size=6)
            ))

            # Predicted Laps (Red Line)
            fig.add_trace(go.Scatter(
                x=clean_df['LapNumber'], 
                y=clean_df['PredictedTime'],
                mode='lines',
                name='AI Prediction',
                line=dict(color='#E10600', width=3)
            ))

            fig.update_layout(
                title=f"Actual vs Predicted Pace ({selected_driver})",
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (s)",
                template="plotly_dark",
                height=500,
                yaxis=dict(autorange="reversed") # In racing, lower is higher on graph usually, or just zoom
            )
            st.plotly_chart(fig, use_container_width=True)

        # ==============================================================================
        # TAB 2: STRATEGY SIMULATOR (Fixed for Driver Specificity)
        # ==============================================================================
        with tab2:
            st.subheader("Strategy Optimizer")
            
            col1, col2 = st.columns(2)
            with col1:
                # Limit slider to actual race length
                max_lap = int(df['LapNumber'].max() - 5)
                current_lap = st.slider("Current Lap", min_value=5, max_value=max_lap, value=20)
            with col2:
                is_sc = st.toggle("Safety Car Deployed?", value=False)
                pit_loss = 11.0 if is_sc else 20.0
                st.caption(f"Pit Loss Cost: **{pit_loss}s**")
            
            # --- COMPOUND SELECTION (FIX #1) ---
            st.markdown("**Tire Strategy:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                current_compound = st.selectbox("Current Tires", ['SOFT', 'MEDIUM', 'HARD'], index=1)
            with c2:
                target_compound = st.selectbox("Pit To", ['SOFT', 'MEDIUM', 'HARD'], index=2)
            with c3:
                # --- WEATHER FORECAST (FIX #3) ---
                current_temp = df['TrackTemp'].iloc[-1] if 'TrackTemp' in df.columns else 35.0
                future_temp = st.slider("Forecast Track Temp (°C)", min_value=15, max_value=55, value=int(current_temp))
                st.caption("Adjust for expected temp change (sunset, clouds)")
            
            st.info(f"Simulating: {selected_driver} ({current_compound} → {target_compound}) | TrackTemp: {future_temp}°C")

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
                
                # --- RESULTS VISUALIZATION ---
                res_df = pd.DataFrame(results)
                
                # Highlight: Calculate Delta to Best
                res_df['Delta'] = res_df['TotalTime'] - res_df['TotalTime'].min()
                best_strat = res_df.loc[res_df['TotalTime'].idxmin()]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🏆 Optimal Box Lap", f"Lap {int(best_strat['PitLap'])}")
                with c2:
                    st.metric("Predicted Race Time", f"{best_strat['TotalTime']:.1f}s")

                fig_strat = go.Figure()
                
                # We plot DELTA instead of raw time to make the curve obvious
                fig_strat.add_trace(go.Scatter(
                    x=res_df['PitLap'],
                    y=res_df['Delta'],
                    mode='lines+markers',
                    name='Time Lost vs Optimal',
                    line=dict(color='#00ff00' if is_sc else '#E10600', width=4),
                    hovertemplate='Pit Lap: %{x}<br>Time Lost: +%{y:.2f}s'
                ))
                
                fig_strat.update_layout(
                    title=f"Strategy Delta: {selected_driver} (Lower is Better)",
                    xaxis_title="Lap You Choose To Pit",
                    yaxis_title="Seconds Slower than Optimal",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig_strat, use_container_width=True)    

        # ==============================================================================
        # FEATURE IMPORTANCE CHART (Bottom of Dashboard)
        # ==============================================================================
        st.markdown("---")
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
                identity_df = identity_df.sort_values('Importance', ascending=False).head(10)
                st.dataframe(identity_df, use_container_width=True, hide_index=True)
                st.caption("These features capture driver/team/track-specific performance differences.")
                
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

if __name__ == "__main__":
    main()