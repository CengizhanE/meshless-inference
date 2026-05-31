import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import json
import time
import os

hide_st_style = """
<style>
/* Safely hide the top-right menu and deploy buttons without breaking the header */
[data-testid="stToolbar"] {visibility: hidden !important;}

/* Hide the standard Streamlit footer */
footer {visibility: hidden;}

/* Add professional margins back to the main UI container */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 4rem !important;
    padding-right: 4rem !important;
}
</style>
"""
def toggle_sidebar():
    """on_click callback — fires BEFORE the rerun so CSS sees the new state."""
    st.session_state.sidebar_open = not st.session_state.sidebar_open

def apply_sidebar_css(is_open: bool):
    base_css = """
    [data-testid="stToolbar"] { visibility: hidden !important; }
    footer { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;  padding-bottom: 2rem !important;
        padding-left: 4rem !important; padding-right: 4rem !important;
    }
    /* Optional: hide the native Streamlit collapse arrow to avoid confusion */
    [data-testid="stSidebarCollapseButton"]  { display: none !important; }
    [data-testid="stSidebarCollapsedControl"]{ display: none !important; }
    """
    sidebar_css = "" if is_open else """
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }
    """
    st.markdown(f"<style>{base_css}{sidebar_css}</style>", unsafe_allow_html=True)

# --- Javascript Injection Hack ---
# This reaches into the browser DOM to manually trigger Streamlit's hidden sidebar button
def force_open_sidebar():
    components.html(
        """
        <script>
        const oldButton = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        const newButton = window.parent.document.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (oldButton) { oldButton.click(); }
        else if (newButton) { newButton.click(); }
        </script>
        """,
        height=0, width=0
    )

# --- Domain/Project Imports ---
from src.models.unet import UNetPhysicsSurrogate

try:
    from src.naca_generator import NACAGenerator
except ImportError:
    st.error("Backend modules not found. Ensure 'src.naca_generator' is in the PYTHONPATH.")


# --- Live Inference Data Generator ---
def generate_inference_tensor(m, p, t, aoa_deg):
    X = np.linspace(-0.5, 1.5, 128)
    Y = np.linspace(-0.5, 0.5, 128)
    grid_x, grid_y = np.meshgrid(X, Y)

    # Inverse Rotation & Mask Math
    aoa_rad = np.radians(aoa_deg)
    grid_x_unrot = grid_x * np.cos(-aoa_rad) - grid_y * np.sin(-aoa_rad)
    grid_y_unrot = grid_x * np.sin(-aoa_rad) + grid_y * np.cos(-aoa_rad)

    # Calculate Thickness Distribution
    yt = 5 * t * (0.2969 * np.sqrt(np.clip(grid_x_unrot, 0, 1)) - 
                  0.1260 * grid_x_unrot - 
                  0.3516 * (grid_x_unrot**2) + 
                  0.2843 * (grid_x_unrot**3) - 
                  0.1015 * (grid_x_unrot**4))
    
    # Calculate Camber Distribution
    yc = np.zeros_like(grid_x_unrot)
    if m > 0:
        front = (grid_x_unrot >= 0) & (grid_x_unrot < p)
        back = (grid_x_unrot >= p) & (grid_x_unrot <= 1)
        yc[front] = (m / p**2) * (2 * p * grid_x_unrot[front] - grid_x_unrot[front]**2)
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * grid_x_unrot[back] - grid_x_unrot[back]**2)

    y_upper = yc + yt
    y_lower = yc - yt

    # Apply Masks
    inside_airfoil = (grid_x_unrot >= 0) & (grid_x_unrot <= 1.0) & (grid_y_unrot <= y_upper) & (grid_y_unrot >= y_lower)
    
    # Generate SDF and AoA Tensors
    binary_mask = np.ones_like(grid_x)
    binary_mask[inside_airfoil] = 0.0
    sdf = distance_transform_edt(binary_mask)
    
    aoa_field = np.full_like(sdf, aoa_deg)
    
    # Shape: (2, 128, 128) -> [SDF_Channel, AoA_Channel]
    input_tensor = np.stack([sdf, aoa_field], axis=0)
    
    return input_tensor, inside_airfoil


# --- Resource Caching ---
@st.cache_resource
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

@st.cache_data
def load_zscores(json_path="dataset/zscore.json"):
    if not os.path.exists(json_path):
        st.error(f"Z-score JSON not found at {json_path}")
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


# --- Main App Configuration ---
def main():
    # 1. MUST be the very first Streamlit call — nothing before this
    st.set_page_config(
        page_title="Meshless.ai | CFD Engine",
        page_icon="meshless_logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Initialize sidebar state
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True

    # 3. Apply CSS immediately — on_click callback has already updated the
    #    state BEFORE this rerun, so this always reads the correct value
    apply_sidebar_css(st.session_state.sidebar_open)

    # 4. Header — replace the if st.button(...) pattern with on_click=
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.title("Meshless.ai - Real-Time Aerodynamics")
    with col2:
        st.write("")
        st.button(
            "⚙️ Toggle Sidebar",
            on_click=toggle_sidebar,      # ← callback, not if-block
            use_container_width=True,
        )
           
    st.markdown("### AI-Driven Surrogate CFD Engine | 12-Drag-Count Accuracy | Millisecond Inference")
    st.divider()

    # --- Sidebar ---
    st.sidebar.header("Airfoil & Flight Parameters")
    st.sidebar.markdown("Adjust the NACA 4-digit profile and Angle of Attack. Inputs are strictly bounded to the model's training distribution.")

    m = st.sidebar.slider("Maximum Camber (m)", min_value=0.0, max_value=0.06, value=0.02, step=0.001, format="%.3f")
    p = st.sidebar.slider("Max Camber Position (p)", min_value=0.2, max_value=0.6, value=0.4, step=0.01, format="%.2f")
    t = st.sidebar.slider("Thickness (t)", min_value=0.08, max_value=0.25, value=0.12, step=0.001, format="%.3f")
    aoa_deg = st.sidebar.slider("Angle of Attack (deg)", min_value=-4.0, max_value=8.0, value=2.0, step=0.1, format="%.1f")
    
    naca_designation = f"NACA {int(m*100)}{int(p*10)}{int(t*100):02d}"
    
    st.sidebar.divider()
    simulate_btn = st.sidebar.button("Simulate Physics", type="primary", use_container_width=True)

    # --- Inference Pipeline ---
    if simulate_btn:
        model, device = load_model()
        zscores = load_zscores()

        if model is None or zscores is None:
            st.stop()

        with st.spinner('Solving Navier-Stokes surrogate...'):
            start_time = time.perf_counter()

            # 1. Generate Input Tensors via the pure math function
            input_tensor_np, inside_airfoil_mask = generate_inference_tensor(m, p, t, aoa_deg)
            
            # Format tensor: (Batch=1, Channels=2, H=128, W=128)
            mask_tensor = torch.tensor(input_tensor_np, dtype=torch.float32).unsqueeze(0).to(device)

            # 2. Model Forward Pass
            with torch.no_grad():
                norm_pressure, norm_coeffs = model(mask_tensor)
                
                # Unpack shape [Batch, 2]
                norm_cl = norm_coeffs[0, 0]
                norm_cd = norm_coeffs[0, 1]

            # 3. Denormalization
            mu_p = zscores['mu']
            sigma_p = zscores['sigma']
            
            # Denormalize pressure field
            pressure_field = (norm_pressure.cpu().numpy()[0, 0, :, :] * sigma_p) + mu_p
            
            cl_pred = norm_cl.cpu().item()
            cd_pred = norm_cd.cpu().item()
            
            # Convert Cd to drag counts
            drag_counts = cd_pred * 10000
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            # --- Main Dashboard: Top Row Metrics ---
            st.subheader(f"Aerodynamic Performance: {naca_designation}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Predicted Lift ($C_l$)", value=f"{cl_pred:.4f}")
            col2.metric(label="Predicted Drag ($C_d$ Counts)", value=f"{drag_counts:.1f} cts", help=f"Absolute Cd: {cd_pred:.5f}")
            col3.metric(label="Inference Time", value=f"{inference_time_ms:.2f} ms")

            # --- Main Dashboard: Bottom Row Visualization ---
            st.divider()
            st.markdown("### Pressure Field Visualization")
            
            masked_pressure = np.ma.masked_where(inside_airfoil_mask, pressure_field)

            fig, ax = plt.subplots(figsize=(12, 6))
            X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 128), np.linspace(-0.5, 0.5, 128))
            
            mesh = ax.pcolormesh(X, Y, masked_pressure, cmap='jet', shading='auto')
            
            generator = NACAGenerator(num_points=200)
            x_coords, y_coords = generator.generate_airfoil(m=m, p=p, t=t)
            aoa_rad_rot = np.radians(-aoa_deg)
            rot_matrix = np.array([[np.cos(aoa_rad_rot), -np.sin(aoa_rad_rot)], 
                                   [np.sin(aoa_rad_rot), np.cos(aoa_rad_rot)]])
            coords = np.vstack((x_coords, y_coords)).T
            rotated_coords = coords.dot(rot_matrix)
            ax.fill(rotated_coords[:, 0], rotated_coords[:, 1], color='black', zorder=2)

            ax.set_aspect('equal')
            ax.set_title(f"Pressure Field (Pa) | {naca_designation} at $\\alpha = {aoa_deg}^\\circ$")
            ax.set_xlabel("X / c")
            ax.set_ylabel("Y / c")
            fig.colorbar(mesh, ax=ax, label="Static Pressure (Pa)")
            
            st.pyplot(fig)
            
    else:
        st.info("👈 Set your airfoil parameters in the sidebar and click **Simulate Physics** to generate predictions.")

if __name__ == "__main__":
    main()
