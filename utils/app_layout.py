import streamlit as st
import os
import signal
import numpy as np
from utils import project_manager

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
            cols = st.columns(5, gap="small")
        else:
            views = ["3D Design", "Preview"]
            cols = st.columns(len(views), gap="small")
            
        for col, view_name in zip(cols, views):
            col.button(view_name, key=f"nav_{view_name}", on_click=set_view, args=(view_name,), 
                       type="primary" if st.session_state.current_view == view_name else "secondary", use_container_width=True)

    with c3:
        st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)
        if st.button("❌ Stop", disabled=not is_maker): st.session_state.quit_confirmation_visible = True

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
        with st.expander("📂 Project", expanded=True):
            # Detect Deployment Environment
            is_web = False
            try:
                if hasattr(st, "secrets") and st.secrets.get("DEPLOYMENT_MODE") == "web":
                    is_web = True
            except Exception: pass
            
            # File Uploader First (to allow updating proj_name)
            if is_maker and is_web:
                uploaded_file = st.file_uploader("📂 Load Project", type=["json"], key="uploader")
                if uploaded_file is not None:
                    import json
                    try:
                        data = json.load(uploaded_file)
                        # Auto-load on upload to avoid button confusion? 
                        # Or use a button with a unique key?
                        if st.button(f"📥 Confirm Load: {uploaded_file.name}", use_container_width=True):
                            project_manager.apply_settings(data)
                            st.session_state.proj_name = data.get('project_name', uploaded_file.name.replace('.json',''))
                            st.toast(f"Loaded {uploaded_file.name}!", icon="📂")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
            
            # Creator Uploader First
            if not is_maker and is_web:
                 creator_upload = st.file_uploader("📂 Load Project to Resume", type=["json"], key="creator_uploader")
                 if creator_upload is not None:
                    import json
                    try:
                        data = json.load(creator_upload)
                        if st.button(f"📥 Restore {creator_upload.name}", use_container_width=True):
                            project_manager.apply_settings(data)
                            st.session_state.proj_name = data.get('project_name', creator_upload.name.replace('.json',''))
                            st.toast(f"Restored {creator_upload.name}!", icon="📂")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Invalid Project File: {e}")
            
            # Name Input (Now safe to render)
            st.text_input("Name", key="proj_name")

            if is_maker:
                if is_web:
                    # WEB MAKER: Download Only (Upload is handled above)
                    json_str, safe_name = project_manager.get_project_json(st.session_state.proj_name, st.session_state)
                    st.download_button("💾 Download Project", data=json_str, file_name=f"{safe_name}.json", mime="application/json", use_container_width=True)
                else:
                    # LOCAL MAKER: Save / Load Disk
                    def save_proj():
                        project_manager.save_project(st.session_state.proj_name, st.session_state)
                        st.toast("Saved!", icon="💾")
                    st.button("💾 Save", on_click=save_proj, use_container_width=True)
                    
                    projs = project_manager.list_projects()
                    if projs:
                        st.selectbox("Load", projs, key="sel_proj", index=None, placeholder="Select a project...")
                        def load_proj():
                            data = project_manager.load_project(st.session_state.sel_proj)
                            if data:
                                project_manager.apply_settings(data)
                                st.session_state.proj_name = st.session_state.sel_proj
                                st.toast(f"Loaded {st.session_state.sel_proj}!", icon="📂")
                        st.button("📂 Load Selected", on_click=load_proj, disabled=not st.session_state.sel_proj, use_container_width=True)
            else:
                # CREATOR MODE
                if is_web:
                    # CREATOR WEB: Load Logic moved to top
                    json_str, safe_name = project_manager.get_project_json(st.session_state.proj_name, st.session_state)
                    st.download_button("💾 Save Project (Download)", data=json_str, file_name=f"{safe_name}.json", mime="application/json", type="secondary", use_container_width=True, help="Download your progress to resume later.")
                    st.download_button("📤 Submit Design (Final)", data=json_str, file_name=f"{safe_name}_SUBMISSION.json", mime="application/json", type="primary", use_container_width=True, help="Download the final design file to email to the Maker.")
                else:
                    if st.button("📤 Submit Design", type="primary", use_container_width=True):
                        st.success("Design Submitted! (Simulation)")

        with st.expander("1. Location", expanded=True):
            st.number_input("Lat", format="%.4f", key="lat"); st.number_input("Lon", format="%.4f", key="lon")
            c1, c2 = st.columns(2); c1.number_input("W (km)", step=1.0, key="width_km"); c2.number_input("H (km)", step=1.0, key="height_km")
        
        with st.expander("2. Box Dimensions", expanded=True):
            st.number_input("Width (mm)", step=1.0, key="box_w"); st.number_input("Height (mm)", step=1.0, key="box_h"); st.number_input("Depth (mm)", step=1.0, key="box_d")
        
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
            
            st.divider()
            
            st.divider()
            st.divider()
        
        # Action Button
        btn_label = "Generate Preview" if not is_maker else "Load/Process Data"
        if not api_key: st.error("No API Key")
        else: st.button(btn_label, type="primary", on_click=process_callback, use_container_width=True)

        # Admin Access (Hidden Toggle)
        st.markdown("---")
        with st.expander("🔒 Admin Access"):
            def check_admin():
                pwd = st.session_state.admin_pwd
                if pwd == "topomaker":
                    st.session_state.user_mode = 'maker'
                elif pwd == "reset":
                    st.session_state.user_mode = 'creator'
                st.session_state.admin_pwd = "" # Clear
                
            st.text_input("Password", type="password", key="admin_pwd", on_change=check_admin, help="Test Password: topomaker")
            if is_maker:
                st.caption("✅ Maker Mode Active")
                if st.button("Logout"):
                    st.session_state.user_mode = 'creator'
                    st.rerun()