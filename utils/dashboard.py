import streamlit as st
import pandas as pd
from utils import project_manager, drive

def render_dashboard():
    """Renders the Maker's Submissions Dashboard."""
    st.title("📬 Submissions Dashboard")
    st.markdown("Review and load designs submitted by Creators.")
    
    # Refresh Button
    # Top Bar: Home | Refresh
    c_back, c_ref, c_spacer = st.columns([1, 1, 4])
    with c_back:
        if st.button("🏠 Home", use_container_width=True):
             st.session_state.current_view = "Home"
             st.rerun()
    with c_ref:
         if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
    # Fetch Data
    with st.spinner("Fetching Submissions..."):
        # We need a way to get detailed info. 
        # list_submissions returns simple names. 
        # For a rich dashboard, we ideally want metadata without downloading everything.
        # But we don't have a database. 
        # Compromise: We list files, and maybe fetch metadata when a row is expanded?
        # Or just show filenames for now, and fetch metadata ON DEMAND.
        
        files = project_manager.list_submissions()
        
    if not files:
        st.info("No submissions found.")
        return

    # Create a DataFrame for the table
    # We will try to parse metadata from the filename if possible, otherwise just list.
    data = []
    for f in files:
        # Check if we have it in drive_map to know source
        is_cloud = False
        if hasattr(st.session_state, 'drive_map') and f in st.session_state.drive_map:
            is_cloud = True
            
        data.append({
            "Project Name": f,
            "Source": "☁️ Cloud" if is_cloud else "💻 Local",
            "Action": f
        })
        
    df = pd.DataFrame(data)
    
    # Display as a grid? Or just a list with columns.
    # We'll use columns for layout control.
    
    # Headers
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.markdown("**Project**")
    c2.markdown("**Source**")
    c3.markdown("**Action**")
    st.divider()
    
    for _, row in df.iterrows():
        c1, c2, c3 = st.columns([3, 1, 1])
        proj_name = row['Project Name']
        
        with c1:
            st.write(f"📂 **{proj_name}**")
            # Expandable details would require loading the JSON.
            # Let's add a "View Details" toggle?
            
        with c2:
            st.write(row['Source'])
            
        with c3:
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
