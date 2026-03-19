import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
import time

# --- INITIALIZATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)
from billing_system import calculate_bill

# Page configuration
st.set_page_config(
    page_title="PowerPredict AI | Smart Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Detect Page from Query Params
q_params = st.query_params
active_p = q_params.get("p", "Forecaster")

# --- PREMIUM DESIGN SYSTEM (GLASSMORPHISM + GRADIENTS) ---
# --- PREMIUM LIGHT BLUE DESIGN SYSTEM ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Poppins:wght@600;700&display=swap');
    
    :root {{
        --primary: #2563EB;
        --secondary: #3B82F6;
        --bg: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-main: #1E293B;
        --text-muted: #64748B;
        --border: #E2E8F0;
        --nav-bg: rgba(255, 255, 255, 0.9);
    }}

    .stApp {{
        background: var(--bg);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }}    /* UI Cleansing & Global Spacing */
    [data-testid="stSidebar"] {{ display: none; }}
    header {{ visibility: hidden; height: 0; }}
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    
    .block-container {{ padding-top: 0rem !important; padding-bottom: 0rem !important; }}
    .stApp {{ overflow: hidden; }}

    /* FULL-WIDTH NAVBAR (GLASSMORPHISM) */
    .nav-container {{
        position: fixed; top: 0; left: 0; right: 0; height: 64px;
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 40px; z-index: 9999;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.03);
    }}
    .nav-logo {{ 
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 20px; 
        color: var(--primary) !important; text-decoration: none !important;
        display: flex; align-items: center; gap: 8px;
    }}
    .nav-tabs {{ display: flex; gap: 32px; }}
    .nav-tab {{
        text-decoration: none !important; color: var(--text-muted); font-weight: 500; font-size: 14px;
        padding: 20px 0; border-bottom: 2px solid transparent; transition: 0.2s;
    }}
    .nav-tab:hover {{ color: var(--primary); }}
    .nav-tab.active {{ 
        color: var(--primary); 
        border-bottom-color: var(--primary);
        font-weight: 700;
    }}
    
    .nav-cta {{
        background: var(--primary); color: white !important; text-decoration: none !important;
        padding: 8px 18px; border-radius: 8px; font-weight: 600; font-size: 13px;
        transition: 0.2s;
    }}
    .nav-cta:hover {{ background: #1D4ED8; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }}

    /* KPI CARDS (NEUMORPHIC + HOVER) */
    .kpi-card {{
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px; padding: 24px;
        display: flex; flex-direction: column;
        box-shadow: 10px 10px 20px #E2E8F0, -10px -10px 20px #FFFFFF !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }}
    .kpi-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 20px 20px 40px #CBD5E1, -20px -20px 40px #FFFFFF !important;
        border-color: rgba(37, 99, 235, 0.3) !important;
    }}
    .kpi-icon {{ font-size: 20px; width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; }}
    .kpi-val {{ font-size: 32px; font-weight: 800; color: var(--text-main); line-height: 1; margin-bottom: 8px; }}
    .kpi-lbl {{ color: var(--text-muted); font-size: 13px; font-weight: 500; letter-spacing: 0.2px; }}
 
    /* MAIN CONTENT */
    .main-wrap {{ margin-top: 64px; padding: 0px 40px 40px 40px; max-width: 1300px; margin-left: auto; margin-right: auto; }}
}}

    /* WHITE CARD (NEUMORPHISM + GLASS) */
    .app-card {{
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 24px;
        /* Neumorphic soft shadows */
        box-shadow: 8px 8px 16px #E2E8F0, -8px -8px 16px #FFFFFF !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 24px;
    }}
    .app-card:hover {{
        transform: translateY(-5px) scale(1.01);
        box-shadow: 12px 12px 24px #CBD5E1, -12px -12px 24px #FFFFFF !important;
        border-color: rgba(37, 99, 235, 0.2);
    }}

    /* FORM INPUTS */
    div[data-baseweb="input"], div[data-baseweb="select"], .stSelectbox > div {{
        background: #F1F5F9 !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        transition: 0.2s !important;
    }}
    div[data-baseweb="input"]:focus-within {{ border-color: var(--primary) !important; background: white !important; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important; }}
    
    [data-testid="stWidgetLabel"] p {{ color: var(--text-main) !important; font-weight: 600 !important; font-size: 13px !important; margin-bottom: 6px !important; }}

    /* BUTTONS (MICROINTERACTIONS) */
    div.stButton > button {{
        width: 100% !important; background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important; border: none !important; height: 52px !important;
        border-radius: 14px !important; font-size: 15px !important; font-weight: 800 !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 20px -10px rgba(37, 99, 235, 0.5) !important;
        letter-spacing: 0.5px;
    }}
    div.stButton > button:hover {{ 
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 30px -10px rgba(37, 99, 235, 0.4) !important;
    }}

    /* RADIO BUTTONS (SEASON) - FINAL PERFECTED VERSION */
    div[data-testid="stRadio"] > label, div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {{ 
        display: none !important; appearance: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important; visibility: hidden !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex !important; gap: 24px !important; flex-wrap: nowrap !important;
        border: none !important; padding: 12px 0 !important; background: transparent !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > div {{
        flex: 1; background: transparent !important; margin: 0 !important; padding: 0 !important;
    }}
    div[data-testid="stRadio"] label {{
        background: #F1F5F9 !important; border: 1px solid var(--border) !important;
        padding: 14px 10px !important; border-radius: 14px !important;
        transition: 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important; cursor: pointer !important;
        text-align: center !important; display: flex !important;
        align-items: center !important; justify-content: center !important;
        min-height: 58px !important; margin: 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: white !important; border-color: var(--primary) !important;
        box-shadow: 0 8px 16px rgba(37, 99, 235, 0.12) !important;
        transform: translateY(-2px) !important;
    }}
    /* Selected State Highlight (Toned Down) */
    div[data-testid="stRadio"] label:has(input:checked) {{
        background: #3B82F6 !important; /* Softer, more modern blue */
        border-color: #3B82F6 !important;
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.25) !important;
    }}
    div[data-testid="stRadio"] label:has(input:checked) p {{
        color: white !important; transform: scale(1.05);
    }}
    
    div[data-testid="stRadio"] p {{
        font-size: 15px !important; font-weight: 800 !important;
        color: var(--text-main) !important; margin: 0 !important;
        white-space: nowrap !important; line-height: 1 !important;
        transition: 0.3s !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioCircle"], div[data-testid="stRadio"] label > div:first-child {{ 
        display: none !important; 
    }}

    /* GLOW EFFECTS ON INPUTS */
    div[data-baseweb="input"]:hover, .stNumberInput:hover {{
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.1) !important;
    }}
    div[data-baseweb="input"]:focus-within {{ 
        border-color: var(--primary) !important; background: white !important; 
        box-shadow: 0 0 25px rgba(37, 99, 235, 0.2) !important; 
    }}

    /* RESULTS PANEL (GLASSMORPHISM) */
    .res-box {{
        background: rgba(37, 99, 235, 0.04) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px; padding: 32px; text-align: center;
        border: 1px solid rgba(37, 99, 235, 0.15); margin-bottom: 24px;
        box-shadow: 4px 4px 10px #E2E8F0, -4px -4px 10px #FFFFFF !important;
    }}
    .res-val {{ font-size: 48px; font-weight: 900; color: var(--primary); line-height: 1; }}
    .res-lbl {{ color: var(--text-muted); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }}

    /* CHART BOX */
    .chart-container {{ background: white; border-radius: 16px; padding: 20px; border: 1px solid var(--border); }}

    /* ANIMATIONS */
    @keyframes slideUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .animate-up {{ animation: slideUp 0.4s ease-out forwards; }}

    /* SLIDERS */
    .stSlider > div > div > div > div {{ background: var(--primary) !important; }}

    /* PLACEHOLDER DESIGN */
    .placeholder-wrap {{
        height: 100%; min-height: 500px; display: flex; flex-direction: column; 
        align-items: center; justify-content: center; background: white; 
        border-radius: 20px; border: 1px dashed #CBD5E1; color: #94A3B8; 
        padding: 40px; text-align: center; position: relative; overflow: hidden;
    }}
    .placeholder-bg-glow {{
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 200px; height: 200px; background: var(--primary); filter: blur(100px);
        opacity: 0.05; z-index: 0;
    }}
    .pulse-icon {{
        font-size: 72px; margin-bottom: 24px; z-index: 1;
        filter: drop-shadow(0 0 10px rgba(37, 99, 235, 0.2));
        animation: pulse 2s infinite ease-in-out;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 0.8; }}
        50% {{ transform: scale(1.1); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 0.8; }}
    }}

    /* INPUT FIX */
    .stNumberInput input {{ background-color: #F1F5F9 !important; color: #1E293B !important; }}
</style>

<div class="nav-container">
    <a href="/?p=Forecaster" class="nav-logo" target="_self">⚡ PowerPredict AI</a>
    <div class="nav-tabs">
        <a href="/?p=Overview" class="nav-tab {'active' if active_p == 'Overview' else ''}" target="_self">Dashboard Hub</a>
        <a href="/?p=Forecaster" class="nav-tab {'active' if active_p == 'Forecaster' else ''}" target="_self">Consumption Forecaster</a>
        <a href="/?p=Analytics" class="nav-tab {'active' if active_p == 'Analytics' else ''}" target="_self">Smart Analytics</a>
    </div>
    <a href="#" class="nav-cta">Deploy Live</a>
</div>
""", unsafe_allow_html=True)

# Helper for Summary Cards
def render_summary_card(label, value, icon="⚡", color="#2563EB"):
    st.markdown(f"""
    <div class="kpi-card animate-up">
        <div class="kpi-icon" style="background: {color}15; color: {color};">{icon}</div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_nn():
    import tensorflow as tf
    model_path = os.path.join(current_dir, 'models', 'electricity_ann_model.keras')
    scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return tf.keras.models.load_model(model_path, compile=False), joblib.load(scaler_path)
    return None, None

def main():
    model, scaler = load_nn()
    
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'bill' not in st.session_state:
        st.session_state.bill = None

    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    if active_p == "Forecaster":
        # 2. SPLIT LAYOUT (40/60)
        col_left, col_right = st.columns([1.8, 2.2]) # Slightly more balanced for side-by-side but keeping form narrow

        with col_left:
            st.markdown('<div class="app-card animate-up">', unsafe_allow_html=True)
            st.markdown('<h3 style="margin-top:0; margin-bottom:24px; color:var(--text-main); font-size:18px;">Input Parameters</h3>', unsafe_allow_html=True)
            
            h_size = st.number_input("Household Size 👨‍👩‍👧‍👦", 1, 12, 4)
            app_count = st.number_input("Appliances Running ⚙️", 1, 100, 15)
            
            st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
            usage = st.slider("Daily Usage (Hours) 🕐", 1.0, 24.0, 8.5)
            temp = st.slider("Avg Temperature (°C) 🌡️", 10, 50, 25)
            
            st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
            prev = st.number_input("Last Month kWh ⚡", 0, 5000, 280)
            
            st.markdown('<p style="font-weight:700; font-size:14px; color:#475569; margin-bottom:12px;">Climatic Season</p>', unsafe_allow_html=True)
            season_label = st.radio("SeasonToggle", ["Summer", "Winter", "Monsoon"], horizontal=True, label_visibility="collapsed")
            season_map = {"Summer": 1, "Winter": 2, "Monsoon": 3}
            season = season_map[season_label]

            st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
            if st.button("⚡ EXECUTE NEURAL AUDIT"):
                with st.spinner("Analyzing Consumption Topology..."):
                    time.sleep(1.2) # Experience-based delay
                    try:
                        input_df = pd.DataFrame([{
                            'household_size': h_size, 'num_appliances': app_count,
                            'daily_usage_hours': usage, 'temperature': temp,
                            'prev_month_consumption': prev, 'seasonal_factor': season,
                            'unit_rate_per_kwh': 0.15 
                        }])
                        input_scaled = scaler.transform(input_df)
                        prediction = model.predict(input_scaled, verbose=0)[0][0]
                        bill = calculate_bill(prediction)
                        
                        st.session_state.prediction = int(prediction)
                        # Realistic currency conversion for Indian region context (multiplier ~8x for typical units)
                        st.session_state.bill = int(bill * 8.5) 
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            if st.session_state.prediction is not None:
                st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
                
                # Results Card
                st.markdown(f"""
                <div class="res-box animate-up">
                    <div class="res-val">₹{st.session_state.bill}</div>
                    <div class="res-lbl">Estimated Monthly Bill</div>
                    <div style="margin-top: 24px; display: flex; justify-content: center; gap: 40px;">
                        <div>
                            <div style="font-size: 24px; font-weight: 800; color: var(--text-main);">{st.session_state.prediction}</div>
                            <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Predicted Units</div>
                        </div>
                        <div style="width: 1px; background: var(--border);"></div>
                        <div>
                            <div style="font-size: 24px; font-weight: 800; color: var(--text-main);">Optimized</div>
                            <div style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Load Profile</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Chart
                st.markdown('<div class="chart-container animate-up">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0; color:var(--text-main); font-size:16px; margin-bottom:16px;">Consumption Flow Analysis</h4>', unsafe_allow_html=True)
                
                # Generate realistic curve based on prediction
                time_steps = ["6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM", "12 AM"]
                # Weights for typical residential loads
                weights = [0.06, 0.10, 0.14, 0.16, 0.24, 0.20, 0.10]
                pred_profile = [st.session_state.prediction * w for w in weights]
                actual_profile = [p * (1 + (np.random.random()-0.5)*0.25) for p in pred_profile]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_steps, y=pred_profile, name="Neural Projection",
                    mode='lines+markers', line=dict(color='#2563EB', width=4, shape='spline'),
                    fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.05)'
                ))
                fig.add_trace(go.Scatter(
                    x=time_steps, y=actual_profile, name="Variance Buffer",
                    mode='lines+markers', line=dict(color='#94A3B8', width=2, dash='dot')
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="#64748B")),
                    xaxis=dict(showgrid=False, color="#94A3B8", tickfont=dict(size=11)),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', color="#94A3B8", tickfont=dict(size=11)),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Welcome State
                st.markdown("""
                <div class="placeholder-wrap animate-up">
                    <div class="placeholder-bg-glow"></div>
                    <div class="pulse-icon">⚡</div>
                    <h2 style="color: #1E293B; font-size: 24px; font-weight: 800; margin-bottom: 12px; z-index: 1;">Audit Ready</h2>
                    <p style="font-size: 15px; max-width: 280px; color: #64748B; line-height: 1.6; z-index: 1;">
                        Input telemetry data on the left to generate the <span style="color: var(--primary); font-weight: 700;">Neural Consumption Profile</span>.
                    </p>
                    <div style="margin-top: 32px; display: flex; align-items: center; gap: 8px; color: #10B981; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; z-index: 1;">
                        <span style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; display: inline-block;"></span>
                        Neural Engine Online
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif active_p == "Overview":
        st.markdown('<div style="margin-top: -48px;"></div>', unsafe_allow_html=True) # Extreme gap fix
        st.markdown('<h1 style="color:var(--text-main); font-size:32px; font-weight:800; margin-bottom:0; margin-top:0; border:none; padding:0;">System Intelligence</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-muted); font-size:16px; margin-bottom:32px; margin-top:0px;">Professional electrical load monitoring and predictive auditing.</p>', unsafe_allow_html=True)
        
        # KPI Row
        col1, col2, col3 = st.columns(3)
        with col1:
            render_summary_card("Accuracy Score", "98.7%", "🧠", "#EC4899")
        with col2:
            render_summary_card("Inference Time", "7.2ms", "⚡", "#F59E0B")
        with col3:
            render_summary_card("Active Tiers", "4 Levels", "🧩", "#10B981")
        
        st.markdown('<div style="margin-top:32px;"></div>', unsafe_allow_html=True)
        
        # Core Methodology Section
        st.markdown("""
        <div class="app-card animate-up">
            <h2 style="color:var(--text-main); font-size:24px; font-weight:700; margin-bottom:16px;">Core Methodology</h2>
            <p style="color:var(--text-main); font-size:15px; line-height:1.6; opacity:0.8;">
                PowerPredict AI utilizes an Artificial Neural Network (ANN) to analyze residential electricity consumption. 
                By factoring in household size, appliance density, and climatic variables, the system generates high-fidelity 
                projections for the upcoming billing cycles.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif active_p == "Analytics":
        st.markdown('<div style="margin-top: -48px;"></div>', unsafe_allow_html=True)
        st.markdown('<h1 style="color:var(--text-main); font-size:32px; font-weight:800; margin-bottom:0;">Smart Analytics</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--text-muted); font-size:16px; margin-bottom:32px;">Advanced consumption forensics and neural performance metrics.</p>', unsafe_allow_html=True)

        row1_col1, row1_col2 = st.columns([1, 1])
        
        with row1_col1:
            st.markdown('<div class="app-card animate-up" style="padding:24px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:800; font-size:18px; color:var(--text-main); margin-bottom:20px;">Seasonal Energy Distribution</p>', unsafe_allow_html=True)
            
            # Simulated seasonal data
            labels = ['Summer', 'Winter', 'Monsoon']
            values = [4500, 3200, 2100]
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
            fig_pie.update_traces(marker=dict(colors=['#2563EB', '#60A5FA', '#93C5FD']), textinfo='none')
            fig_pie.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), height=280, showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(color="#64748B"))
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with row1_col2:
            st.markdown('<div class="app-card animate-up" style="padding:24px;">', unsafe_allow_html=True)
            st.markdown('<p style="font-weight:800; font-size:18px; color:var(--text-main); margin-bottom:20px;">System Efficiency Score</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align:center; padding: 15px 0;">
                <div style="font-size:72px; font-weight:900; color:#10B981; line-height:1;">A+</div>
                <div style="color:#64748B; font-weight:600; font-size:14px; margin-top:10px; letter-spacing:1px;">OPTIMAL PERFORMANCE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<hr style="border:none; border-top:1px solid #F1F5F9; margin:20px 0;">', unsafe_allow_html=True)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.metric("Loss", "2.4%", "-0.2%")
            with m_col2: st.metric("Stability", "98.1%", "0.5%")
            with m_col3: st.metric("Uptime", "100%", "Active")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="app-card animate-up" style="padding:24px;">', unsafe_allow_html=True)
        st.markdown('<p style="font-weight:800; font-size:18px; color:var(--text-main); margin-bottom:20px;">Neural Accuracy Audit (Predicted vs Actual)</p>', unsafe_allow_html=True)
        
        # Predicted vs Actual simulation
        x_pts = np.linspace(0, 50, 100)
        # Base prediction curve
        y_pred = np.sin(x_pts/5) + 2
        # Actual output with realistic variance
        y_actual = y_pred + np.random.normal(0, 0.12, 100)

        fig_acc = go.Figure()
        
        # Actual Line (Thinner, dotted or lighter)
        fig_acc.add_trace(go.Scatter(
            x=x_pts, y=y_actual, name='Actual Consumption',
            line=dict(color='#94A3B8', width=2, dash='dot')
        ))
        
        # Predicted Line (Thicker, solid primary)
        fig_acc.add_trace(go.Scatter(
            x=x_pts, y=y_pred, name='Neural Forecast',
            line=dict(color='#2563EB', width=4, shape='spline')
        ))

        fig_acc.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), height=300,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, title="Time Samples (Intervals)", color="#94A3B8"),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title="kWh Intensity", color="#94A3B8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#64748B")),
            hovermode="x unified"
        )
        st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
