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
    
    # Progress bar if many files? Local read is fast, but let's be safe.
    # We'll just read them.
    for f in files:
        # Load the full file to get metadata (Fast local read)
        # We can implement a lightweight peeker later if needed.
        json_data = project_manager.load_submission(f) 
        
        # Defaults
        author = "Anonymous"
        email = "-"
        time_str = "-"
        note = ""
        dims = "-"
        
        if json_data:
            # Meta
            info = json_data.get('submission_info', {})
            author = info.get('author') or "Anonymous"
            email = info.get('email') or "-"
            time_str = info.get('timestamp', '-').strip()
            note = info.get('comments', '')
            
            # Dims
            box = json_data.get('box', {})
            w = box.get('width', 0)
            h = box.get('height', 0)
            dims = f"{w:.0f} x {h:.0f} mm"

        data.append({
            "Project": f,
            "Date": time_str,
            "User": author,
            "Dimensions": dims,
            "Email": email,
            "Note": note,
            "raw_name": f # hidden key
        })
        
    df = pd.DataFrame(data)
    
    # Sort by Date (descending) ideally, but string date is unreliable.
    # Let's trust string sort or file mtime in future.
    
    # Headers
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    c1.markdown("**Project**")
    # c2: Date
    c2.markdown("**Date**")
    # c3: User
    c3.markdown("**User**")
    # c4: Dims
    c4.markdown("**Size**")
    # c5: Action
    c5.markdown("**Action**")
    st.divider()
    
    for _, row in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
        proj_name = row['raw_name']
        
        with c1:
            st.write(f"📂 **{proj_name}**")
            if row['Note']:
                st.caption(f"📝 {row['Note']}")
        
        with c2:
            st.write(row['Date'])
            
        with c3:
            st.write(f"**{row['User']}**")
            if row['Email'] != "-":
                st.caption(row['Email'])
                
        with c4:
            st.write(row['Dimensions'])
            
        with c5:
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
