import streamlit as st
import pandas as pd
import subprocess
from utils import project_manager

def render_dashboard():
    """Renders the Maker's Submissions Dashboard."""
    st.title("📬 Submissions Dashboard")
    st.markdown("Review and load designs submitted by Creators (Synced via GitHub).")
    
    # Refresh Button
    # Top Bar: Home | Refresh
    c_back, c_ref, c_spacer = st.columns([1, 1, 4])
    with c_back:
        if st.button("🏠 Home", use_container_width=True):
             st.session_state.current_view = "Home"
             st.rerun()
    with c_ref:
         if st.button("🔄 Sync (Git Pull)", use_container_width=True):
            try:
                # Run git pull to fetch latest submissions
                subprocess.run(["git", "pull"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                st.toast("Synced with Cloud! 🐙")
            except Exception as e:
                st.error(f"Sync error: {e}")
                
            st.cache_data.clear()
            st.rerun()
        
    # Fetch Data
    with st.spinner("Loading Submissions..."):
        files = project_manager.list_submissions()
        
    if not files:
        st.info("No submissions found (Try Syncing).")
        return

    # Create a DataFrame for the table
    data = []
    for f in files:
        data.append({
            "Project Name": f,
            "Action": f
        })
        
    df = pd.DataFrame(data)
    
    # Headers
    c1, c2 = st.columns([3, 1])
    c1.markdown("**Project**")
    c2.markdown("**Action**")
    st.divider()
    
    for _, row in df.iterrows():
        c1, c2 = st.columns([3, 1])
        proj_name = row['Project Name']
        
        with c1:
            st.write(f"📂 **{proj_name}**")
            
        with c2:
            st.button("⬇️ Load", key=f"load_{proj_name}", on_click=_load_and_switch, args=(proj_name,), use_container_width=True)

def _load_and_switch(proj_name):
    """Loads the project and switches back to the Studio tab."""
    # Use the unified loader
    data = project_manager.load_submission(proj_name)
    if data:
        project_manager.apply_settings(data)
        st.session_state.proj_name = data.get('project_name', proj_name)
        st.toast(f"Loaded {proj_name}!", icon="📂")
        
        # Check for submission Info to display
        if 'submission_info' in data:
            info = data['submission_info']
            st.session_state.last_submission_info = info # Store to show in Studio
            
        # Switch Tab (We need to handle navigation state in app.py)
        st.session_state.current_view = "3D Design"  # Fixed view name
        # No rerun needed here if using on_click, Streamlit reruns automatically after callback
    else:
        st.error("Failed to load project.")
