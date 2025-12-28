import streamlit as st
import copy
from utils import views 

def render_assembly_tools_archived(cur_idx, n_total, final_geoms, current_dowels):
    """
    ARCHIVED 2025-12-27
    Former implementation of Assembly Tools with Manual Edits and History limits.
    """
    # views._check_slider_sync() # Internal function, commented out
    layer_key = f"L{cur_idx}"
    current_layer_polys = final_geoms[cur_idx - 1]
    # views._handle_click_capture(f"assembly_plot_L{cur_idx}", f"last_click_assembly_L{cur_idx}", layer_key)
    
    jig_mods_to_render = copy.deepcopy(st.session_state.jig_modifications.get(layer_key, []))

    # Layout: Tools Left, Vis Right
    c_tools, c_vis = st.columns([1, 3])
    
    with c_tools:
        st.markdown("##### Layer Control")
        col_prev, col_next = st.columns(2)
        # col_prev.button("◀ Prev", ...) 
        # col_next.button("Next ▶", ...)
        
        # def on_slider_change():
        #     st.session_state.layer_idx = st.session_state.layer_idx_slider_assembly
        #     views._sync_sliders()
        # st.slider("Layer", 1, n_total, key="layer_idx_slider_assembly_OLD", on_change=on_slider_change, label_visibility="collapsed")
        st.divider()

        t_settings, t_manual, t_history = st.tabs(["Settings", "Manual", "History"])
        with t_settings:
            # st.number_input("Strut Width", ...) REMOVED per user request (Auto-calculated)
            st.info("Mode: Grid Supports (Material Efficient, Grid-Aligned).")
            # st.checkbox("Fluid Smoothing", value=True, key="jig_fluid_OLD", help="Applies organic fillets to intersections for a 'molded' look.")

        with t_manual:
            mj_c1, mj_c2 = st.columns(2)
            mj_x = mj_c1.number_input("Center X", key=f"jig_x_L{cur_idx}_OLD")
            mj_y = mj_c2.number_input("Center Y", key=f"jig_y_L{cur_idx}_OLD")
            st.session_state.setdefault(f"jig_w_L{cur_idx}", 20.0)
            st.session_state.setdefault(f"jig_h_L{cur_idx}", 20.0)
            mj_c3, mj_c4 = st.columns(2)
            mj_w = mj_c3.number_input("W", step=5.0, key=f"jig_w_L{cur_idx}_OLD")
            mj_h = mj_c4.number_input("H", step=5.0, key=f"jig_h_L{cur_idx}_OLD")
            
            def add_mod(m_type):
                st.session_state.jig_modifications.setdefault(layer_key, [])
                st.session_state.jig_modifications[layer_key].append({'type': m_type, 'x': mj_x, 'y': mj_y, 'w': mj_w, 'h': mj_h})
            mj_b1, mj_b2 = st.columns(2)
            mj_b1.button("➕ Add", on_click=add_mod, args=('add',), use_container_width=True, key="add_jig_OLD")
            mj_b2.button("✂️ Cut", on_click=add_mod, args=('sub',), use_container_width=True, key="sub_jig_OLD")

        with t_history:
            mods = st.session_state.jig_modifications.get(layer_key, [])
            if mods:
                sel_jig_idx = st.radio("Edit", range(len(mods)), format_func=lambda i: f"#{i+1} {mods[i]['type'].title()}", key=f"jig_sel_{cur_idx}_OLD", horizontal=True)
                
                if sel_jig_idx is not None and sel_jig_idx < len(mods):
                    if st.button("🗑️ Delete", key=f"del_jig_mod_btn_{cur_idx}_OLD", use_container_width=True):
                        st.session_state.jig_modifications[layer_key].pop(sel_jig_idx)
                        st.rerun()

        st.divider()
        # View Options
        show_dowels = st.checkbox("Show Dowels", value=True, key="show_dowels_ui_OLD")

    with c_vis:
        # Render Assembly
        pass # views._render_2d_assembly_view(...)


def render_dowel_ui_archived():
    """
    ARCHIVED 2025-12-27
    Dowel Positioning Logic from Sidebar.
    """
    st.checkbox("Dowels", key="use_dowels_OLD")
    if st.session_state.get('use_dowels_OLD'):
        st.number_input("Diam (mm)", step=1.0, key="dowel_diam_OLD"); st.number_input("Count", 1, 4, key="num_dowels_OLD")
        # Logic for snapping dowels etc...
