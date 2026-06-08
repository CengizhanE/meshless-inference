import streamlit as st
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
import json
import time
import os
import pandas as pd
import plotly.graph_objects as go

# --- Domain/Project Imports ---
from src.models.unet import UNetPhysicsSurrogate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.naca_generator import NACAGenerator
except ImportError:
    st.error("Backend modules not found. Ensure 'src.naca_generator' is in the PYTHONPATH.")


# --- UI & Styling Configuration ---
def toggle_sidebar():
    """on_click callback — fires BEFORE the rerun so CSS sees the new state."""
    st.session_state.sidebar_open = not st.session_state.sidebar_open

def inject_custom_css(is_sidebar_open: bool):
    sidebar_css = "" if is_sidebar_open else """
    section[data-testid="stSidebar"] { display: none !important; visibility: hidden !important; }
    """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Apply Inter to the whole app */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Apply JetBrains Mono ONLY to metric numbers and inference time */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.5px;
    }}

    /* 1. Base Streamlit Cleanups */
    [data-testid="stToolbar"] {{ visibility: hidden !important; }}
    footer {{ visibility: hidden; }}
    .block-container {{
        max-width: 95% !important;     
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;  
        padding-right: 2rem !important;   
    }}
    [data-testid="stSidebarCollapseButton"]  {{ display: none !important; }}
    [data-testid="stSidebarCollapsedControl"]{{ display: none !important; }}
    
    {sidebar_css}

    /* 2. Tech-Style Metric Cards */
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(30, 34, 40, 1) 0%, rgba(14, 17, 23, 1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 75, 75, 0.3); 
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        border-top: 1px solid rgba(255, 75, 75, 0.8);
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.15);
    }}
    [data-testid="stMetricLabel"] p {{
        color: #8b949e !important; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85rem;
    }}

    /* 3. "Coming Soon" Styling */
    [data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] > div,
    [data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] > div {{
        font-size: 1.3rem !important; 
        color: rgba(255, 255, 255, 0.2) !important; 
        font-style: italic; 
        font-weight: 400 !important;
        letter-spacing: 1px;
        margin-top: 5px;
    }}

    /* 4. Glowing Primary Button */
    [data-testid="baseButton-primary"] {{
        background-color: #ff4b4b !important;
        border: none !important;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.4) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }}
    [data-testid="baseButton-primary"]:hover {{
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.8) !important;
        transform: scale(1.02);
    }}

    /* 4.5. High-Tech Secondary/Download Button */
    [data-testid="baseButton-secondary"] {{
        background-color: transparent !important;
        border: 1px solid rgba(255, 75, 75, 0.5) !important;
        color: #ff4b4b !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }}
    [data-testid="baseButton-secondary"]:hover {{
        background-color: rgba(255, 75, 75, 0.1) !important;
        border-color: #ff4b4b !important;
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.2) !important;
        transform: translateY(-2px);
    }}

    /* 5. Subtle Background Gradient */
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at 50% 0%, #1e222b 0%, #0e1117 40%, #000000 100%) !important;
    }}
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    /* 6. Hide Streamlit's Default Header Anchor Links */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {{
        display: none !important;
    }}

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
        margin: 2rem 0;
    }}

    /* --- Sidebar Typography --- */
    /* Force Inter explicitly on all text elements inside the sidebar */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Make the slider numbers techy/monospace to match your metrics */
    div[data-testid="stThumbValue"], 
    div[data-testid="stTickBarMin"], 
    div[data-testid="stTickBarMax"] {{
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.5px;
        color: #8b949e !important; /* Muted grey so they don't distract */
    }}
            
    </style>
    """, unsafe_allow_html=True)


# --- Live Inference Data Generator ---
def generate_inference_tensor(m, p, t, aoa_deg):
    X = np.linspace(-0.5, 1.5, 128)
    Y = np.linspace(-0.5, 0.5, 128)
    grid_x, grid_y = np.meshgrid(X, Y)

    aoa_rad = np.radians(aoa_deg)
    grid_x_unrot = grid_x * np.cos(-aoa_rad) - grid_y * np.sin(-aoa_rad)
    grid_y_unrot = grid_x * np.sin(-aoa_rad) + grid_y * np.cos(-aoa_rad)

    yt = 5 * t * (0.2969 * np.sqrt(np.clip(grid_x_unrot, 0, 1)) - 
                  0.1260 * grid_x_unrot - 
                  0.3516 * (grid_x_unrot**2) + 
                  0.2843 * (grid_x_unrot**3) - 
                  0.1015 * (grid_x_unrot**4))
    
    yc = np.zeros_like(grid_x_unrot)
    if m > 0:
        front = (grid_x_unrot >= 0) & (grid_x_unrot < p)
        back = (grid_x_unrot >= p) & (grid_x_unrot <= 1)
        yc[front] = (m / p**2) * (2 * p * grid_x_unrot[front] - grid_x_unrot[front]**2)
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * grid_x_unrot[back] - grid_x_unrot[back]**2)

    y_upper = yc + yt
    y_lower = yc - yt

    inside_airfoil = (grid_x_unrot >= 0) & (grid_x_unrot <= 1.0) & (grid_y_unrot <= y_upper) & (grid_y_unrot >= y_lower)
    
    binary_mask = np.ones_like(grid_x)
    binary_mask[inside_airfoil] = 0.0
    sdf = distance_transform_edt(binary_mask)
    
    aoa_field = np.full_like(sdf, aoa_deg)
    input_tensor = np.stack([sdf, aoa_field], axis=0)
    
    return input_tensor, inside_airfoil


# --- Resource Caching ---
@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path="checkpoints/unet_resnet_epoch_300.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint not found at {checkpoint_path}")
        return None, device
    
    try:
        model = UNetPhysicsSurrogate() 
        state_dict = torch.load(checkpoint_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

@st.cache_data(show_spinner=False)
def load_zscores(json_path="dataset/zscore.json"):
    if not os.path.exists(json_path):
        st.error(f"Z-score JSON not found at {json_path}")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


# --- Main App Configuration ---
def main():
    st.set_page_config(
        page_title="Meshless.ai | CFD Engine",
        page_icon="https://raw.githubusercontent.com/CengizhanE/meshless-inference/main/meshless_logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. State Initialization
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
        
    # This is the memory for the simulation!
    if "simulated" not in st.session_state:
        st.session_state.simulated = False

    # Inject all CSS
    inject_custom_css(st.session_state.sidebar_open)

    # Pre-Warm the models silently in the background
    _, _ = load_model()
    _ = load_zscores()

    # Header
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.title("Meshless.ai - Real-Time Aerodynamics")
    with col2:
        st.write("")
        st.button("Toggle Sidebar", on_click=toggle_sidebar, use_container_width=True)
            
    st.markdown("##### AI-Driven Surrogate CFD Engine | Millisecond Inference")
    st.divider()

    # --- Sidebar ---
    st.sidebar.header("Airfoil & Flight Parameters")
    st.sidebar.markdown("Adjust the NACA profile and AoA. Inputs are bounded to training distribution.")

    m = st.sidebar.slider("Maximum Camber (m)", 0.0, 0.06, 0.02, 0.001, "%.3f", 
                          help="Maximum camber in percentage of the chord (e.g., 0.02 = 2%).")
    p = st.sidebar.slider("Max Camber Position (p)", 0.2, 0.6, 0.4, 0.01, "%.2f", 
                          help="Position of maximum camber in tenths of the chord (e.g., 0.4 = 40%).")
    t = st.sidebar.slider("Thickness (t)", 0.08, 0.25, 0.12, 0.001, "%.3f", 
                          help="Maximum thickness of the airfoil as a fraction of the chord.")
    aoa_deg = st.sidebar.slider("Angle of Attack (deg)", -4.0, 8.0, 2.0, 0.1, "%.1f",
                          help="The angle between the oncoming air and the chord line of the airfoil.")
    
    naca_designation = f"NACA {int(m*100)}{int(p*10)}{int(t*100):02d}"
    
    st.sidebar.divider()
    st.sidebar.markdown("**Visualization Settings**")
    plot_theme = st.sidebar.radio("Graph Theme", ["Dark", "Light"], horizontal=True)
    
    st.sidebar.divider()
    
    # 2. Update the Memory when Button is Clicked
    simulate_btn = st.sidebar.button("Simulate Physics", type="primary", use_container_width=True)
    if simulate_btn:
        st.session_state.simulated = True

    # --- Inference Pipeline ---
    if st.session_state.simulated:
        model, device = load_model()
        zscores = load_zscores()

        if model is None or zscores is None:
            st.stop()

        with st.spinner('Solving Navier-Stokes surrogate...'):
            start_time = time.perf_counter()

            input_tensor_np, inside_airfoil_mask = generate_inference_tensor(m, p, t, aoa_deg)
            mask_tensor = torch.tensor(input_tensor_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                norm_pressure, norm_coeffs = model(mask_tensor)

            # Denormalization
            mu_p = zscores['mu']
            sigma_p = zscores['sigma']
            pressure_field = (norm_pressure.cpu().numpy()[0, 0, :, :] * sigma_p) + mu_p
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            # --- Dashboard Top Row ---
            st.subheader(f"Aerodynamic Performance: {naca_designation}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Predicted Lift ($C_l$)", value="Coming Soon")
            col2.metric(label="Predicted Drag ($C_d$ Counts)", value="Coming Soon") 
            col3.metric(label="Inference Time", value=f"{inference_time_ms:.2f} ms")

            # --- Dashboard Visualization (PLOTLY VERSION) ---
            st.divider()
            st.markdown("### Pressure Field Visualization")
            
            # --- Dynamic Graph Theme Setup for Plotly ---
            if plot_theme == "Dark":
                fig_bg = 'rgba(0,0,0,0)' # Transparent so Streamlit gradient shows
                ax_bg = '#0e1117'
                airfoil_fill = '#000000'
                airfoil_edge = 'white'
                text_color = '#ffffff'
                grid_color = '#2b303b'
            else:
                fig_bg = '#ffffff'
                ax_bg = '#ffffff'
                airfoil_fill = '#333333'
                airfoil_edge = 'black'
                text_color = '#000000'
                grid_color = '#e6e6e6'

            # Plotly prefers np.nan for masked areas to render them transparently
            pressure_for_plotly = np.where(inside_airfoil_mask, np.nan, pressure_field)
            
            # 1D axes for Plotly
            x_ax = np.linspace(-0.5, 1.5, 128)
            y_ax = np.linspace(-0.5, 0.5, 128)

            fig = go.Figure()

            # 1. The Pressure Heatmap
            fig.add_trace(go.Heatmap(
                z=pressure_for_plotly,
                x=x_ax,
                y=y_ax,
                colorscale='Turbo',
                colorbar=dict(
                    title=dict(
                        text="Static Pressure (Pa)",
                        side="right",
                        font=dict(color=text_color, family="Inter")
                    ),
                    tickfont=dict(color=text_color, family="JetBrains Mono")
                ),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Pressure: %{z:.1f} Pa<extra></extra>"
            ))

            # 2. The Airfoil Shape Generation
            generator = NACAGenerator(num_points=200)
            x_coords, y_coords = generator.generate_airfoil(m=m, p=p, t=t)
            aoa_rad_rot = np.radians(-aoa_deg)
            rot_matrix = np.array([[np.cos(aoa_rad_rot), -np.sin(aoa_rad_rot)], 
                                   [np.sin(aoa_rad_rot), np.cos(aoa_rad_rot)]])
            coords = np.vstack((x_coords, y_coords)).T
            rotated_coords = coords.dot(rot_matrix)

            # 3. Add the Airfoil to Plotly
            fig.add_trace(go.Scatter(
                x=rotated_coords[:, 0],
                y=rotated_coords[:, 1],
                fill="toself",
                fillcolor=airfoil_fill,
                line=dict(color=airfoil_edge, width=2),
                hoverinfo="skip", # Disables the tooltip over the black airfoil interior
                showlegend=False
            ))

            # 4. Layout Configuration
            fig.update_layout(
                title=dict(
                    text=f"Static Pressure Field (Pa) | {naca_designation} at α = {aoa_deg}°",
                    font=dict(color=text_color, family="Inter", size=18)
                ),
                xaxis=dict(
                    title="X / c", 
                    color=text_color, 
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    range=[-0.2, 1.2], 
                    constrain="domain"
                ),
                yaxis=dict(
                    title="Y / c", 
                    color=text_color, 
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    scaleanchor="x", 
                    scaleratio=1,
                    range=[-0.4, 0.4],
                    constrain="domain"
                ),
                plot_bgcolor=ax_bg,
                paper_bgcolor=fig_bg,
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode="closest",
                font=dict(family="Inter")
            )

            # Render Plotly in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True) # Adds a tiny bit of spacing
            
            coord_df = pd.DataFrame(rotated_coords, columns=["X", "Y"])
            csv_data = coord_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="🠋  Download Airfoil Coordinates (CSV)",
                data=csv_data,
                file_name=f"{naca_designation}_coords.csv",
                mime="text/csv",
                use_container_width=True
            )

            # --- Custom Footer ---
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem; margin-bottom: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.05);">
                <p style="color: rgba(255, 255, 255, 0.3); font-size: 0.85rem; font-style: italic; letter-spacing: 0.5px;">
                    Simulation successfully rendered in <strong style="color: #ffffff;">{0:.2f} ms</strong>. 
                    <br>Powered by <span style="color: #ff4b4b; font-weight: 600; font-style: normal;">Meshless.ai</span> Surrogate Physics Engine.
                </p>
            </div>
            """.format(inference_time_ms), unsafe_allow_html=True)
            
    else:
        # --- Custom Empty State Prompt with SVG Arrow ---
        st.markdown("""
        <div style="
            background: linear-gradient(145deg, rgba(30, 34, 40, 0.7) 0%, rgba(14, 17, 23, 0.7) 100%);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-left: 4px solid #ff4b4b;
            padding: 24px 30px;
            border-radius: 8px;
            color: #8b949e;
            font-size: 1.05rem;
            font-weight: 500;
            margin-top: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            letter-spacing: 0.5px;
            display: flex; 
            align-items: center; 
            justify-content: center;
            gap: 12px;
        ">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ff4b4b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <line x1="19" y1="12" x2="5" y2="12"></line>
                <polyline points="12 19 5 12 12 5"></polyline>
            </svg>
            <span>Set your airfoil parameters in the sidebar and click <strong style="color: #ffffff;">Simulate Physics</strong> to generate predictions.</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()