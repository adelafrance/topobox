import streamlit as st
import os
import time
from utils import project_manager

def render_home():
    """Renders the Landing Page/Home Screen."""
    
    st.title("🏔️ TopoBox")
    st.caption("Advanced Topographic Model Generator")
    
    # Maker "Stop" Button (Top Right)
    if st.session_state.get('user_mode') == 'maker':
        # Absolute positioning hack or just columns? Columns is safer.
        # We want the title to be left, and stop button to be right.
        # But st.title takes a whole line usually.
        # Let's use a sidebar or just place it below title? 
        # Actually, let's inject it into a column layout at the top.
        pass # We'll do it below.

    # Quit Confirmation (Home Screen Version)
    import signal
    if st.session_state.get('quit_home_visible', False):
         with st.container(border=True):
            st.warning("Are you sure you want to quit?")
            def do_quit(): 
                os.kill(os.getpid(), signal.SIGTERM)
            def do_cancel(): 
                st.session_state.quit_home_visible = False
            
            b1, b2 = st.columns(2)
            b1.button("🏃 Quit App", on_click=do_quit, type="primary", use_container_width=True)
            b2.button("↩️ Cancel", on_click=do_cancel, use_container_width=True)
            
    if st.session_state.get('user_mode') == 'maker':
         if st.button("❌ Stop App", key="home_stop"):
             st.session_state.quit_home_visible = True
             st.rerun()

    st.divider()
    
    # Three main entry points
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.container(border=True).markdown("#### ✨ New Project")
        st.write("Start a new design from scratch.")
        new_name = st.text_input("Project Name", placeholder="e.g., Alps_01", key="home_new_name")
        
        if st.button("Create Project", type="primary", use_container_width=True, disabled=not new_name):
            # Initialize New State
            st.session_state.proj_name = new_name
            # Reset core vars to trigger defaults in engine
            keys_to_reset = ['lat', 'lon', 'width_km', 'height_km', 'box_w', 'box_h', 'box_d']
            for k in keys_to_reset:
                if k in st.session_state: del st.session_state[k]
                
            st.session_state.current_view = "Studio"
            st.rerun()
            
    with c2:
        st.container(border=True).markdown("#### 📂 Resume Project")
        st.write("Continue working on a local save.")
        
        projects = project_manager.list_projects()
        selected = st.selectbox("Select Project", options=projects, key="home_load_select")
        
        if st.button("Load Project", type="secondary", use_container_width=True, disabled=not selected):
            try:
                data = project_manager.load_project(selected)
                if data:
                    # Apply settings
                    # project_manager.apply_settings updates st.session_state in-place
                    project_manager.apply_settings(data)
                    st.session_state.proj_name = selected
                    st.session_state.current_view = "Studio"
                    st.rerun()
                else:
                    st.error("Failed to load project data.")
            except Exception as e:
                st.error(f"Error loading project: {e}")

    if st.session_state.get('user_mode') == 'maker':
        with c3:
            st.container(border=True).markdown("#### 👷 Maker Dashboard")
            st.write("View and print submitted designs.")
            if st.button("Open Dashboard", use_container_width=True):
                st.session_state.current_view = "Dashboard"
                st.rerun()

    st.divider()
    
    # Recent Status / Tips (Optional)
    with st.expander("ℹ️  Quick Tips", expanded=False):
        tips = """
        - **New Project**: Starts with default Switzerland coordinates.
        - **Resume**: Loads `.json` files from your `projects/` folder.
        """
        if st.session_state.get('user_mode') == 'maker':
            tips += "- **Dashboard**: Checks the Cloud Database for team submissions."
        
        st.markdown(tips)
