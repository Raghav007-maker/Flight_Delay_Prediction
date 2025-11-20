import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils import *
from utils import check_project_files
import os

def map_airline_code_to_name(airline_code):
    """Convert API airline codes to encoder format"""
    mapping = {
        'AA': 'American Airlines Inc.',
        'DL': 'Delta Air Lines Inc.',
        'UA': 'United Air Lines Inc.',
        'WN': 'Southwest Airlines Co.',
        'B6': 'JetBlue Airways',
        'AS': 'Alaska Airlines Inc.',
        'NK': 'Spirit Air Lines',
        'F9': 'Frontier Airlines Inc.',
        'G4': 'Allegiant Air',
        'HA': 'Hawaiian Airlines Inc.'
    }
    return mapping.get(airline_code, None)

def map_callsign_to_airline(callsign):
    """Map OpenSky callsign to airline name"""
    mapping = {
        'AAL': 'American Airlines Inc.',
        'DAL': 'Delta Air Lines Inc.',
        'UAL': 'United Air Lines Inc.',
        'SWA': 'Southwest Airlines Co.',
        'JBU': 'JetBlue Airways',
        'ASA': 'Alaska Airlines Inc.',
        'NKS': 'Spirit Air Lines',
        'FFT': 'Frontier Airlines Inc.',
        'AAY': 'Allegiant Air',
        'HAL': 'Hawaiian Airlines Inc.'
    }
    
    for prefix, airline in mapping.items():
        if callsign.startswith(prefix):
            return airline, prefix[:2]
    
    return 'Unknown Airline', 'N/A'

st.set_page_config(
    page_title="US Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; margin: 1rem 0;}
    h1 {color: #1f77b4; padding-bottom: 0.5rem; border-bottom: 2px solid #1f77b4;}
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'api_calls' not in st.session_state:
    st.session_state.api_calls = 0
if 'cached_predictions' not in st.session_state:
    st.session_state.cached_predictions = None
if 'cache_time' not in st.session_state:
    st.session_state.cache_time = None

st.title("✈️ US Flight Delay Probability Prediction System")
st.markdown("**Powered by XGBoost ML | Trained on 300K US Flights (2022)**")

# Check files
files_ok, missing_files = check_project_files()

if not files_ok:
    st.error("⚠️ **Setup Incomplete!** Missing required files:")
    for file in missing_files:
        st.error(f"  ❌ {file}")
    st.stop()

# Load models
with st.spinner("Loading ML models..."):
    model, airline_encoder, origin_encoder, feature_names, error = safe_load_models()

if error:
    st.error(f"❌ **Error loading models:** {error}")
    st.stop()

st.success("✅ Models loaded successfully!")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# API Source Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Data Source")

api_source = st.sidebar.radio(
    "Choose Flight Data API:",
    ["AviationStack", "OpenSky Network (Free & Real-time)"],
    help="OpenSky provides real-time data without API key!"
)

# API Settings based on selection
with st.sidebar.expander("🔑 API Settings", expanded=True):
    if api_source == "AviationStack":
        API_KEY = st.text_input(
            "AviationStack API Key", 
            type="password",
            help="Enter your API key from aviationstack.com"
        )
        
        if API_KEY:
            is_valid, message = validate_api_key(API_KEY)
            if is_valid:
                st.success("✅ API key format valid")
            else:
                st.warning(f"⚠️ {message}")
        
        st.caption(f"API calls this session: {st.session_state.api_calls}/100")
        st.info("⚠️ Free tier: 100 calls/month, updates every 60 min")
        
    else:  # OpenSky Network
        st.success("✅ Using OpenSky Network")
        st.info("📡 No API key needed!")
        st.caption(f"API calls this session: {st.session_state.api_calls}")
        st.success("✨ Real-time data (10-second updates)")
        st.success("🎯 4000 calls/day limit")
        API_KEY = None  # Not needed for OpenSky

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
st.sidebar.metric("Accuracy", "79.62%")
st.sidebar.metric("ROC-AUC", "0.85+")
st.sidebar.metric("Training Data", "300K flights")

st.sidebar.markdown("---")
st.sidebar.caption(f"Total predictions: {st.session_state.prediction_count}")

# Main tabs
tab1, tab2, tab3 = st.tabs([
    "🔴 Live Predictions", 
    "🎯 Single Flight", 
    "ℹ️ About"
])

def extract_features_from_api(flight_data):
    """Extract features from API flight data (AviationStack format)"""
    try:
        scheduled_dep = flight_data['departure']['scheduled']
        dep_datetime = datetime.fromisoformat(scheduled_dep.replace('Z', '+00:00'))
        
        airline_code = flight_data['airline']['iata']
        airline_full_name = map_airline_code_to_name(airline_code)
        
        if airline_full_name is None:
            return None
        
        origin = flight_data['departure']['iata']
        
        features = {
            'Month': dep_datetime.month,
            'DayOfWeek': dep_datetime.weekday(),
            'DepHour': dep_datetime.hour,
            'IsWeekend': 1 if dep_datetime.weekday() >= 5 else 0,
            'Distance': 500,
            'Airline': airline_full_name,
            'Origin': origin
        }
        
        return features
        
    except Exception as e:
        return None

def encode_features(features):
    """Encode categorical features"""
    try:
        airline = features['Airline']
        origin = features['Origin']
        
        if airline not in airline_encoder.classes_:
            return None
        
        features['Airline_Encoded'] = airline_encoder.transform([airline])[0]
        
        if origin not in origin_encoder.classes_:
            origin = 'OTHER'
        
        features['Origin_Encoded'] = origin_encoder.transform([origin])[0]
        
        features.pop('Airline')
        features.pop('Origin')
        
        return features
        
    except Exception as e:
        return None

def predict_delay(flight_data):
    """Predict delay for a single flight"""
    features = extract_features_from_api(flight_data)
    if features is None:
        return None
    
    features = encode_features(features)
    if features is None:
        return None
    
    feature_array = [[
        features['Month'],
        features['DayOfWeek'],
        features['DepHour'],
        features['IsWeekend'],
        features['Distance'],
        features['Airline_Encoded'],
        features['Origin_Encoded']
    ]]
    
    try:
        probability = model.predict_proba(feature_array)[0][1]
        prediction = model.predict(feature_array)[0]
        risk, color, status = get_risk_level(probability)
        
        return {
            'probability': probability,
            'prediction': 'DELAYED' if prediction == 1 else 'ON-TIME',
            'risk_level': risk,
            'risk_color': color,
            'risk_status': status
        }
    except Exception as e:
        return None

def get_live_flights(api_key, limit=50):
    """Fetch live flights from AviationStack"""
    try:
        url = "http://api.aviationstack.com/v1/flights"
        
        params = {
            'access_key': api_key,
            'limit': 100,
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.api_calls += 1
            
            all_flights = data.get('data', [])
            
            if not all_flights:
                return None
            
            # US airports
            us_airports = [
                'LAX', 'JFK', 'ORD', 'ATL', 'DFW', 'DEN', 'SFO', 'LAS', 'PHX', 'IAH',
                'CLT', 'MCO', 'SEA', 'EWR', 'BOS', 'MSP', 'DTW', 'PHL', 'LGA', 'FLL',
                'BWI', 'DCA', 'MDW', 'SAN', 'TPA', 'PDX', 'STL', 'HOU', 'OAK', 'AUS'
            ]
            
            us_airlines = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'G4', 'HA']
            
            # Filter US flights
            us_flights = []
            for flight in all_flights:
                try:
                    origin = flight.get('departure', {}).get('iata', '')
                    dest = flight.get('arrival', {}).get('iata', '')
                    airline = flight.get('airline', {}).get('iata', '')
                    
                    if origin in us_airports or dest in us_airports or airline in us_airlines:
                        us_flights.append(flight)
                        
                except Exception:
                    continue
            
            if us_flights:
                return us_flights[:limit]
            else:
                return None
            
        else:
            return None
            
    except Exception as e:
        return None

def get_live_flights_opensky(limit=50):
    """Fetch live flights from OpenSky Network (FREE, Real-time)"""
    try:
        url = "https://opensky-network.org/api/states/all"
        
        st.write("🌐 Fetching from OpenSky Network...")
        
        response = requests.get(url, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.api_calls += 1
            
            if not data or 'states' not in data or not data['states']:
                st.warning("⚠️ No flights data returned from OpenSky")
                return None
            
            flights_raw = data['states']
            st.write(f"✅ Fetched {len(flights_raw)} flights worldwide")
            
            # Parse OpenSky format
            flights = []
            
            for state in flights_raw:
                try:
                    # Skip flights on ground
                    if state[8]:  # on_ground flag
                        continue
                    
                    callsign = str(state[1]).strip() if state[1] else 'N/A'
                    origin_country = state[2] if state[2] else 'Unknown'
                    altitude = state[7] if state[7] else 0
                    velocity = state[9] if state[9] else 0
                    
                    is_us_flight = False
                    
                    # Check if from United States
                    if origin_country == "United States":
                        is_us_flight = True
                    
                    # Check if callsign starts with US airline codes
                    us_airline_prefixes = ['AAL', 'DAL', 'UAL', 'SWA', 'JBU', 'ASA', 'NKS', 'FFT', 'AAY', 'HAL']
                    for prefix in us_airline_prefixes:
                        if callsign.startswith(prefix):
                            is_us_flight = True
                            break
                    
                    if is_us_flight and altitude and altitude > 1000:  # Only cruising flights
                        airline_name, airline_code = map_callsign_to_airline(callsign)
                        
                        # Convert to AviationStack-like format
                        flights.append({
                            'flight': {'iata': callsign},
                            'airline': {
                                'name': airline_name,
                                'iata': airline_code
                            },
                            'departure': {
                                'iata': 'N/A',  # OpenSky doesn't provide origin
                                'scheduled': datetime.now().isoformat()
                            },
                            'arrival': {'iata': 'N/A'},
                            'flight_status': 'active',  # All OpenSky flights are active
                            'opensky_data': {
                                'altitude': altitude,
                                'velocity': velocity,
                                'country': origin_country
                            }
                        })
                
                except Exception:
                    continue
            
            st.write(f"🇺🇸 US flights found: {len(flights)}")
            
            if flights:
                return flights[:limit]
            else:
                st.warning("⚠️ No US flights found in OpenSky data")
                return None
        
        elif response.status_code == 429:
            st.error("❌ Rate limit exceeded on OpenSky API")
            return None
        else:
            st.error(f"❌ OpenSky API returned status: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"❌ Error fetching from OpenSky: {e}")
        return None

# TAB 1: LIVE PREDICTIONS
with tab1:
    st.header("🔴 Live Flight Delay Predictions")
    
    if api_source == "OpenSky Network (Free & Real-time)":
        st.markdown("**Analyze real-time US domestic flights using OpenSky Network** 🌐")
        st.info("✨ Real-time data with 10-second updates | No API key required | 4000 calls/day")
    else:
        st.markdown("**Analyze real-time US domestic flights - Summary View Only**")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        num_flights = st.number_input("Number of flights", min_value=10, max_value=100, value=30, step=10)
    
    with col2:
        fetch_button = st.button("🔄 Fetch Flights", type="primary", use_container_width=True)
    
    # Check cache
    cache_valid = False
    if st.session_state.cache_time:
        time_diff = (datetime.now() - st.session_state.cache_time).total_seconds()
        if time_diff < 300:  # 5 minutes cache
            cache_valid = True
            st.info(f"📦 Using cached data from {st.session_state.cache_time.strftime('%H:%M:%S')} ({int(300-time_diff)}s remaining)")
    
    if fetch_button or (cache_valid and st.session_state.cached_predictions is not None):
        # Check if API key needed
        if api_source == "AviationStack" and not API_KEY and not cache_valid:
            st.error("⚠️ Please enter AviationStack API key in sidebar")
        else:
            # Use cache or fetch new
            if cache_valid and st.session_state.cached_predictions is not None and not fetch_button:
                results = st.session_state.cached_predictions
                st.success(f"✅ Loaded {len(results)} flights from cache")
            else:
                with st.spinner("🌐 Fetching live flights..."):
                    # Choose API based on selection
                    if api_source == "AviationStack":
                        flights = get_live_flights(API_KEY, limit=num_flights)
                    else:
                        flights = get_live_flights_opensky(limit=num_flights)
                
                if not flights:
                    st.error("❌ Failed to fetch flights. Try again.")
                    st.stop()
                
                st.success(f"✅ Fetched {len(flights)} flights")
                
                # Process flights silently
                results = []
                progress_bar = st.progress(0)
                
                for i, flight in enumerate(flights):
                    try:
                        airline_code = flight['airline'].get('iata', 'N/A')
                        us_airline_codes = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9', 'G4']
                        
                        if airline_code in us_airline_codes or api_source == "OpenSky Network (Free & Real-time)":
                            prediction = predict_delay(flight)
                            
                            if prediction:
                                flight_status = flight.get('flight_status', 'unknown').upper()
    
                                # Add status emoji
                                status_emoji = {
                                    'ACTIVE': '✅',
                                    'SCHEDULED': '⏰',
                                    'LANDED': '🛬',
                                    'CANCELLED': '❌',
                                    'UNKNOWN': '❓'
                                }.get(flight_status, '❓')
                                
                                results.append({
                                    'Flight': flight['flight'].get('iata', 'N/A'),
                                    'Airline': flight['airline'].get('name', 'Unknown'),
                                    'Route': f"{flight['departure'].get('iata', 'N/A')} → {flight['arrival'].get('iata', 'N/A')}",
                                    'Scheduled': flight['departure'].get('scheduled', 'N/A')[:16],
                                    'Status': f"{status_emoji} {flight_status}",
                                    'Delay_Probability': prediction['probability'],
                                    'Delay_Prob_%': f"{prediction['probability']*100:.1f}%",
                                    'Risk_Level': prediction['risk_level'],
                                    'Prediction': prediction['prediction']
                                })
                                
                                st.session_state.prediction_count += 1
                        
                        progress_bar.progress((i + 1) / len(flights))
                    except:
                        continue
                
                progress_bar.empty()
                
                # Cache results
                st.session_state.cached_predictions = results
                st.session_state.cache_time = datetime.now()
            
            if results:
                df_results = pd.DataFrame(results)
                
                # === SUMMARY SECTION ===
                st.markdown("---")
                st.markdown("## 📊 Summary")
                
                total_flights = len(df_results)
                high_risk = len(df_results[df_results['Risk_Level'] == '🔴 HIGH'])
                medium_risk = len(df_results[df_results['Risk_Level'] == '🟡 MEDIUM'])
                low_risk = len(df_results[df_results['Risk_Level'] == '🟢 LOW'])
                avg_delay_prob = df_results['Delay_Probability'].mean()
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Flights", total_flights)
                
                with col2:
                    st.metric("High Risk", high_risk, 
                             delta=f"{(high_risk/total_flights*100):.1f}%", 
                             delta_color="inverse")
                
                with col3:
                    st.metric("Medium Risk", medium_risk, 
                             delta=f"{(medium_risk/total_flights*100):.0f}%")
                
                with col4:
                    st.metric("Low Risk", low_risk, 
                             delta=f"{(low_risk/total_flights*100):.0f}%", 
                             delta_color="normal")
                
                # Average delay banner
                st.markdown(f"""
                <div style='background: #4A5568; padding: 1.5rem; border-radius: 10px; 
                            text-align: center; margin: 1.5rem 0; color: white;
                            border: 2px solid #667eea;'>
                    <h2 style='margin: 0; font-size: 1.8rem;'>📈 Average Delay Probability: {avg_delay_prob*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # === FLIGHT DETAILS TABLE ===
                st.markdown("---")
                st.markdown("## ✈️ Flight Details")
                
                st.info(f"📊 Showing {len(df_results)} flights from predictions")
                
                # Filter and sort options
                filter_col1, filter_col2 = st.columns([3, 1])
                
                unique_risks = df_results['Risk_Level'].unique().tolist()
                
                with filter_col1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Level:",
                        unique_risks,
                        default=unique_risks
                    )

                    unique_statuses = df_results['Status'].unique().tolist()
                    status_filter = st.multiselect(
                        "Filter by Flight Status:",
                        unique_statuses,
                        default=unique_statuses
                    )
                
                with filter_col2:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Highest Risk", "Lowest Risk", "Flight Number", "Airline"]
                    )
                
                # Apply filters
                if risk_filter:
                    filtered_df = df_results[df_results['Risk_Level'].isin(risk_filter)].copy()
                else:
                    filtered_df = df_results.copy()
                
                if status_filter:
                    filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)].copy()
                
                # Apply sorting
                if sort_option == "Highest Risk":
                    filtered_df = filtered_df.sort_values('Delay_Probability', ascending=False)
                elif sort_option == "Lowest Risk":
                    filtered_df = filtered_df.sort_values('Delay_Probability', ascending=True)
                elif sort_option == "Flight Number":
                    filtered_df = filtered_df.sort_values('Flight')
                else:
                    filtered_df = filtered_df.sort_values('Airline')
                
                # Display table
                st.dataframe(
                    filtered_df[['Flight', 'Airline', 'Route', 'Scheduled', 'Status', 'Delay_Prob_%', 'Risk_Level', 'Prediction']],
                    use_container_width=True,
                    height=400
                )
                
                # Download buttons
                st.markdown("### 📥 Download Options")
                
                download_col1, download_col2, download_col3 = st.columns(3)
                
                with download_col1:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        csv,
                        f"flight_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with download_col2:
                    summary_text = f"""
Flight Delay Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {api_source}

=== SUMMARY ===
Total Flights Analyzed: {total_flights}
High Risk: {high_risk} ({high_risk/total_flights*100:.1f}%)
Medium Risk: {medium_risk} ({medium_risk/total_flights*100:.1f}%)
Low Risk: {low_risk} ({low_risk/total_flights*100:.1f}%)

Average Delay Probability: {avg_delay_prob*100:.1f}%

=== TOP 5 AIRLINES ===
{df_results['Airline'].value_counts().head(5).to_string()}

Model: XGBoost | Accuracy: 79.62% | ROC-AUC: 0.85+
                    """
                    
                    st.download_button(
                        "📊 Download Summary",
                        summary_text,
                        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                with download_col3:
                    full_csv = df_results.to_csv(index=False)
                    st.download_button(
                        "📑 Download Full Data",
                        full_csv,
                        f"full_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                # === VISUALIZATIONS ===
                st.markdown("---")
                st.markdown("## 📊 Visual Analytics")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'],
                        values=[low_risk, medium_risk, high_risk],
                        marker=dict(colors=['#4CAF50', '#FFC107', '#F44336']),
                        hole=0.4
                    )])
                    fig_pie.update_layout(
                        title="Risk Distribution",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    airline_counts = df_results['Airline'].value_counts().head(5)
                    fig_bar = px.bar(
                        x=airline_counts.values,
                        y=airline_counts.index,
                        orientation='h',
                        title="Top 5 Airlines by Flight Count",
                        labels={'x': 'Flights', 'y': 'Airline'},
                        color=airline_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig_bar.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("### Delay Probability Distribution")
                fig_hist = px.histogram(
                    df_results,
                    x='Delay_Probability',
                    nbins=20,
                    title="Distribution of Delay Probabilities Across All Flights",
                    labels={'Delay_Probability': 'Delay Probability'},
                    color_discrete_sequence=['#667eea']
                )
                fig_hist.add_vline(x=avg_delay_prob, line_dash="dash", 
                                  line_color="red", 
                                  annotation_text=f"Avg: {avg_delay_prob*100:.1f}%")
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.markdown("### Average Delay Probability by Airline")
                airline_avg = df_results.groupby('Airline')['Delay_Probability'].mean().sort_values(ascending=False)
                fig_airline = px.bar(
                    x=airline_avg.values * 100,
                    y=airline_avg.index,
                    orientation='h',
                    title="Which Airlines Have Higher Delay Risk?",
                    labels={'x': 'Average Delay Probability (%)', 'y': 'Airline'},
                    color=airline_avg.values * 100,
                    color_continuous_scale='RdYlGn_r'
                )
                fig_airline.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_airline, use_container_width=True)
                
            else:
                st.warning("⚠️ No valid predictions. Try again with different flights.")

# TAB 2: SINGLE FLIGHT
with tab2:
    st.header("🎯 Single Flight Prediction")
    st.markdown("Enter flight details manually for custom prediction")
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.subheader("📅 Date & Time")
        
        flight_date = st.date_input(
            "Flight Date",
            value=datetime.now(),
            min_value=datetime.now(),
            max_value=datetime.now() + timedelta(days=365)
        )
        
        month = flight_date.month
        day_of_week_num = flight_date.weekday()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week = day_names[day_of_week_num]
        
        st.info(f"📅 {day_of_week}, {flight_date.strftime('%B %d, %Y')}")
        
        dep_time = st.time_input("Departure Time", value=datetime.now().time())
        dep_hour = dep_time.hour
        
        st.info(f"⏰ Hour: {dep_hour}:00")
        
    with input_col2:
        st.subheader("✈️ Flight Details")
        
        us_airlines = ['American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.', 
                      'Southwest Airlines Co.', 'JetBlue Airways', 'Alaska Airlines Inc.']
        
        available_airlines = [a for a in us_airlines if a in airline_encoder.classes_]
        airline = st.selectbox("Airline", available_airlines)
        
        major_us_airports = ['LAX', 'JFK', 'ORD', 'ATL', 'DFW', 'DEN', 'SFO', 'LAS', 'PHX', 'IAH']
        available_origins = [o for o in major_us_airports if o in origin_encoder.classes_]
        origin = st.selectbox("Origin Airport", available_origins)
        
        distance = st.number_input("Distance (miles)", 100, 5000, 1000, 50)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        is_weekend = 1 if day_of_week_num >= 5 else 0
        airline_encoded = airline_encoder.transform([airline])[0]
        origin_encoded = origin_encoder.transform([origin])[0]
        
        features = [[month, day_of_week_num, dep_hour, is_weekend, distance, airline_encoded, origin_encoded]]
        
        with st.spinner("🔮 Analyzing..."):
            probability = model.predict_proba(features)[0][1]
            prediction = model.predict(features)[0]
        
        st.session_state.prediction_count += 1
        
        risk, risk_color, risk_status = get_risk_level(probability)
        
        st.markdown(f"""
        <div class='prediction-card'>
            <h1 style='font-size: 5rem; margin: 0;'>{probability*100:.1f}%</h1>
            <h3 style='margin-top: 0.5rem;'>Delay Probability</h3>
            <p style='margin-top: 1rem; font-size: 1.2rem;'>Risk: <strong>{risk}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", ['ON-TIME', 'DELAYED'][prediction])
        
        with col2:
            st.metric("Confidence", f"{max(probability, 1-probability)*100:.1f}%")
        
        with col3:
            expected_delay = int(probability * 45) if prediction == 1 else 0
            st.metric("Expected Delay", f"{expected_delay} min")

# TAB 3: ABOUT
with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## ✈️ Flight Delay Probability Prediction System
    
    A machine learning system that predicts flight delay probabilities using:
    - **XGBoost ML Model** trained on 300K flights
    - **Real-time Flight Data** from AviationStack or OpenSky Network APIs
    - **Advanced Analytics** for risk assessment
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - Live flight tracking
        - Delay probability
        - Risk categorization
        - Visual analytics
        - Data export options
        - Dual API support
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Performance
        - Accuracy: 79.62%
        - ROC-AUC: 0.85+
        - 300K training samples
        - US domestic focus
        """)
    
    with col3:
        st.markdown("""
        ### 🛠️ Tech Stack
        - Python & Streamlit
        - XGBoost ML
        - Plotly Charts
        - AviationStack API
        - OpenSky Network API
        """)
    
    st.markdown("---")
    st.subheader("🌐 API Comparison")
    
    comparison_data = {
        'Feature': ['API Key Required', 'Monthly Limit', 'Real-time Updates', 'Data Freshness', 'Coverage', 'Best For'],
        'AviationStack': ['✅ Yes', '100 calls/month', '❌ 60 min delay', 'Moderate', 'Global', 'Testing'],
        'OpenSky Network': ['❌ No', '4000 calls/day', '✅ 10 seconds', 'Excellent', 'Global', 'Production']
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.info("💡 **Recommendation:** Use OpenSky Network for real-time, non-repeating flight data!")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption(f"⚡ Predictions: {st.session_state.prediction_count}")

with footer_col2:
    if api_source == "AviationStack":
        st.caption(f"📡 API Calls: {st.session_state.api_calls}/100")
    else:
        st.caption(f"📡 API Calls: {st.session_state.api_calls}/4000 today")

with footer_col3:
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")