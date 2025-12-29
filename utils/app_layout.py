import streamlit as st
import os
import signal
import numpy as np
from utils import project_manager
from streamlit_javascript import st_javascript

def inject_custom_css():
    st.markdown("""
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; }
            h1 { padding-top: 0rem; margin-top: 0.5rem; }
            button[kind="primary"] { background-color: #2196F3 !important; border-color: #2196F3 !important; color: white !important; }
            button[kind="primary"]:hover { background-color: #1976D2 !important; border-color: #1976D2 !important; color: white !important; }
            div.stButton > button { padding: 0rem 0.25rem; min-height: 0px; height: auto; }
            div.stButton > button > div > p { white_space: nowrap !important; }
        </style>
    """, unsafe_allow_html=True)

def render_navigation():
    # Role-Based View Filtering
    is_maker = st.session_state.get('user_mode') == 'maker'
    
    c1, c2, c3 = st.columns([2, 6, 1], gap="small")
    with c1: 
        st.title("🏔️ TopoBox")
        if is_maker: st.caption("Maker Mode")
    
    def set_view(view): st.session_state.current_view = view

    with c2:
        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
        
        # Define Views based on Role
        if is_maker:
            views = ["3D Design", "Layer Tools", "Assembly Tools", "Preview", "Export"]
            cols = st.columns(len(views), gap="small")
        else:
            views = ["3D Design", "Preview"]
            cols = st.columns(len(views), gap="small")
            
        for col, view_name in zip(cols, views):
            col.button(view_name, key=f"nav_{view_name}", on_click=set_view, args=(view_name,), 
                       type="primary" if st.session_state.current_view == view_name else "secondary", use_container_width=True)

    with c3:
        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
        # Stop Button only for Makers
        if is_maker and st.button("❌ Stop"): st.session_state.quit_confirmation_visible = True

    if st.session_state.get('quit_confirmation_visible', False):
        with st.container(border=True):
            st.warning("Are you sure you want to quit?")
            def do_save_and_quit(): 
                project_manager.save_project(st.session_state.proj_name, st.session_state)
                os.kill(os.getpid(), signal.SIGTERM)
            def do_quit(): os.kill(os.getpid(), signal.SIGTERM)
            def do_cancel(): st.session_state.quit_confirmation_visible = False
            b1, b2, b3 = st.columns(3)
            b1.button("💾 Save & Quit", on_click=do_save_and_quit, type="primary", use_container_width=True)
            b2.button("🏃 Quit", on_click=do_quit, use_container_width=True)
            b3.button("↩️ Cancel", on_click=do_cancel, use_container_width=True)

def render_sidebar(api_key, process_callback):
    is_maker = st.session_state.get('user_mode') == 'maker'
    
    with st.sidebar:
        # NEW Home Button
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = "Home"
            st.rerun()
            
        with st.expander("❓ How to Use", expanded=False):
            st.markdown("""
            1. **Locate:** Use the **📍** button or enter coordinates to find your mountain.
            2. **Size:** Set the **Map Width** (terrain size) and **Box Dimensions** (wood size).
            3. **Preview:** Click **Generate Preview** to see the 3D result.
            4. **Save:** Download the `.json` file and send it to your maker!
            """)
        
        with st.expander("📂 Project", expanded=True):
            # Name Input 
            st.text_input("Name", key="proj_name")

            # Detect Deployment Environment
            is_web = False
            try:
                if hasattr(st, "secrets") and st.secrets.get("DEPLOYMENT_MODE") == "web":
                    is_web = True
            except Exception: pass

            if is_maker:
                if is_web:
                    # WEB MAKER: Download Only
                    json_str, safe_name = project_manager.get_project_json(st.session_state.proj_name, st.session_state)
                    st.download_button("💾 Download Project", data=json_str, file_name=f"{safe_name}.json", mime="application/json", use_container_width=True)
                else:
                    # LOCAL MAKER: Save to Disk
                    def save_proj():
                        project_manager.save_project(st.session_state.proj_name, st.session_state)
                        st.toast("Saved!", icon="💾")
                    st.button("💾 Save", on_click=save_proj, use_container_width=True)
                    
            else:
                # CREATOR MODE
                st.markdown("### Submission")
                with st.expander("📝 Details (Optional)", expanded=True):
                    sub_name = st.text_input("Your Name", key="sub_author")
                    sub_email = st.text_input("Email", key="sub_email")
                    sub_note = st.text_area("Comments", key="sub_note", height=80)
                
                submission_meta = {
                    "author": sub_name,
                    "email": sub_email,
                    "comments": sub_note,
                    "timestamp": str(os.popen("date").read()).strip() # Simple timestamp
                }

                if is_web:
                    # CREATOR WEB: 
                    json_str, safe_name = project_manager.get_project_json(st.session_state.proj_name, st.session_state)
                    st.download_button("💾 Download", data=json_str, file_name=f"{safe_name}.json", mime="application/json", type="secondary", use_container_width=True)
                    
                    # Cloud Submission
                    if st.button("📤 Submit (Cloud)", type="primary", use_container_width=True, key=f"btn_sub_{safe_name}", disabled=not sub_name):
                         try:
                             with st.status("Processing...", expanded=True) as status:
                                 status.write("📦 Packaging...")
                                 status.write("☁️ Uploading (Updating DB)...")
                                 fname, err = project_manager.submit_design(st.session_state.proj_name, st.session_state, submission_meta)
                                 
                                 if err:
                                     status.write(f"⚠️ Cloud Error: {err}")
                                     status.update(label="Saved Locally", state="error", expanded=True)
                                 else:
                                     status.update(label="Done!", state="complete", expanded=False)
                                 
                             if err:
                                 st.warning(f"Saved Locally Only. ({err})")
                             else:
                                 st.toast(f"Submitted!", icon="✅")
                                 st.success(f"Submitted as '{fname}'!")
                         except Exception as e:
                             st.error(f"Failed: {str(e)}")
                else:
                    if st.button("📤 Submit Design", type="primary", use_container_width=True):
                        # FIX: Unpack tuple
                        fname, err = project_manager.submit_design(st.session_state.proj_name, st.session_state, submission_meta)
                        if err:
                             st.warning(f"Saved Locally (Cloud Error: {err})")
                        else:
                             st.toast(f"Submitted!", icon="✅")
                             st.success(f"Submitted to Cloud as '{fname}'!")


        with st.expander("2. Map Settings & Dimensions", expanded=True):
            # Auto-populate text input
            if 'coords_input' not in st.session_state:
                st.session_state.coords_input = f"{st.session_state.lat:.4f}, {st.session_state.lon:.4f}"
            
            def _parse_coords(txt):
                try:
                    parts = txt.split(',')
                    if len(parts) == 2:
                        st.session_state.lat = float(parts[0].strip())
                        st.session_state.lon = float(parts[1].strip())
                except:
                    pass

            # --- Geolocation Logic (Must run BEFORE widget instantiation) ---
            if st.session_state.get('getting_loc'):
                with st.status("Requesting Location...", expanded=True) as status:
                    st.write("Please 'Allow' browser permissions.")
                    if st.button("Cancel", key="cancel_geo"):
                        st.session_state.getting_loc = False
                        st.rerun()
                        
                    # Run JS Geolocation
                    js_code = """new Promise(resolve => {
                        navigator.geolocation.getCurrentPosition(
                            pos => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                            err => resolve({error: err.message}),
                            {enableHighAccuracy: true, timeout: 8000, maximumAge: 0}
                        );
                    })"""
                    
                    loc_data = st_javascript(js_code, key=st.session_state.get('geo_key', 'geo_init'))
                    
                    if loc_data and loc_data != 0:
                        if isinstance(loc_data, dict) and 'error' in loc_data:
                             status.write(f"Error: {loc_data['error']}")
                             status.update(label="Failed", state="error")
                             time.sleep(2)
                             st.session_state.getting_loc = False
                             st.rerun()
                        elif isinstance(loc_data, dict) and 'lat' in loc_data:
                            st.session_state.lat = loc_data['lat']
                            st.session_state.lon = loc_data['lon']
                            # Update state BEFORE widget renders
                            st.session_state.coords_input = f"{loc_data['lat']:.4f}, {loc_data['lon']:.4f}"
                            st.session_state.getting_loc = False
                            status.write("Found!")
                            status.update(label="Success", state="complete")
                            st.rerun()
                        else:
                             # Wait / No Data yet
                             pass
            # ----------------------------------------------------------------

            c_txt, c_btn = st.columns([0.85, 0.15])
            with c_txt:
                st.text_input("Lat/Lon", key="coords_input", on_change=lambda: _parse_coords(st.session_state.coords_input), help="Format: Lat, Lon", label_visibility="collapsed")
            with c_btn:
                if st.button("📍", help="Get Current Location", use_container_width=True):
                    import time
                    st.session_state.getting_loc = True
                    st.session_state.geo_key = f"geo_{time.time()}"
                    st.rerun()

            st.caption(f"Current: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
            
            # Aspect Ratio Logic
            lock_aspect = st.checkbox("🔒 Lock Aspect to Box", value=True, key="lock_aspect")
            
            def update_h_from_w():
                if st.session_state.lock_aspect and st.session_state.box_w > 0 and st.session_state.box_h > 0:
                    ratio = st.session_state.box_h / st.session_state.box_w
                    st.session_state.height_km = st.session_state.width_km * ratio
            
            def update_w_from_h():
                 if st.session_state.lock_aspect and st.session_state.box_w > 0 and st.session_state.box_h > 0:
                    ratio = st.session_state.box_w / st.session_state.box_h
                    st.session_state.width_km = st.session_state.height_km * ratio
            
            def update_dims_from_box():
                 if st.session_state.lock_aspect:
                     update_h_from_w()

            c1, c2 = st.columns(2)
            c1.number_input("W (km)", step=0.5, key="width_km", on_change=update_h_from_w, min_value=1.0, max_value=25.0, help="Width of the real-world terrain to capture.")
            c2.number_input("H (km)", step=0.5, key="height_km", on_change=update_w_from_h, min_value=1.0, max_value=25.0, help="Height of the real-world terrain to capture.")
            
            st.caption("Physical Box Dimensions:")
            c_b1, c_b2, c_b3 = st.columns(3)
            c_b1.number_input("W (mm)", step=10.0, key="box_w", min_value=10.0, on_change=update_dims_from_box, help="Interior Width of your laser-cut box.")
            c_b2.number_input("H (mm)", step=10.0, key="box_h", min_value=10.0, on_change=update_dims_from_box, help="Interior Height of your laser-cut box.")
            c_b3.number_input("D (mm)", step=5.0, key="box_d", min_value=5.0, help="Interior Depth (available stacking height).")
        
        with st.expander("3. Material", expanded=True):
            st.number_input("Thickness (mm)", min_value=0.1, step=0.1, key="mat_th", help="Thickness of the material sheet.")
            if st.session_state.get('show_adjustment_info', False): st.info(f"Adjusted: **{st.session_state.adjusted_mat_th:.2f} mm**")
        
        with st.expander("4. Frame", expanded=True):
            st.selectbox("Style", ["None", "Full Perimeter", "Custom Sides"], key="frame_mode")
            if st.session_state.frame_mode != "None":
                st.number_input("Width (mm)", min_value=1.0, step=1.0, key="frame_width")
                if st.session_state.frame_mode == "Custom Sides":
                    c1, c2 = st.columns(2)
                    st.session_state.frame_sides['top'] = c1.checkbox("Top", value=st.session_state.frame_sides['top'])
                    st.session_state.frame_sides['left'] = c1.checkbox("Left", value=st.session_state.frame_sides['left'])
                    st.session_state.frame_sides['bottom'] = c2.checkbox("Bottom", value=st.session_state.frame_sides['bottom'])
                    st.session_state.frame_sides['right'] = c2.checkbox("Right", value=st.session_state.frame_sides['right'])

        if is_maker:
            with st.expander("5. Process (Advanced)", expanded=False):
                c_b1, c_b2 = st.columns([3, 1])
                st.session_state.blur_slider = st.session_state.blur
                st.session_state.blur_input = st.session_state.blur
                def update_blur_slider(): st.session_state.blur = st.session_state.blur_slider
                def update_blur_input(): st.session_state.blur = st.session_state.blur_input
                with c_b1: st.slider("Blur", 0.0, 10.0, key="blur_slider", on_change=update_blur_slider)
                with c_b2: st.number_input("Blur", 0.0, 10.0, key="blur_input", on_change=update_blur_input, label_visibility="collapsed", step=0.1)
                
                st.number_input("Min Area (mm²)", step=1.0, key="min_area")
                st.number_input("Min Feature Width (mm)", min_value=0.5, step=0.1, key="min_feature_width")
                st.divider()
                st.markdown("**Intelligent Automation**")
                st.checkbox("Auto-Connect Nearby Islands", value=True, key="auto_bridge", help="Automatically bridges islands that are very close to the mainland (within 1.5x bridge thickness).")
                st.checkbox("Auto-Remove Raster Noise", value=True, key="auto_cleanup", help="Automatically removes extremely tiny specs (< 0.5x width) which are usually just rasterization noise.")
                if st.checkbox("Auto-Fuse Touching Parts", value=True, key="auto_fuse", help="Merges features that are essentially touching into single solid objects."):
                    st.number_input("Max Merge Gap (mm)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="fuse_gap", help="Any parts closer than this distance will be fused together.")
            

        # Action Button
        btn_label = "Generate Preview" if not is_maker else "Process / Update"
        if not api_key: st.error("No API Key")
        else: st.button(btn_label, type="primary", on_click=process_callback, use_container_width=True)