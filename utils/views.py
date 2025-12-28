import streamlit as st
import plotly.graph_objects as go
import numpy as np
import copy
import base64
import math
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from shapely import affinity
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import to_rgba
from utils import visualizer, geometry_engine, exporter, nesting

def set_cam(eye, up, view='design'): 
    st.session_state[f'camera_snap_{view}'] = dict(eye=eye, up=up)

def dec_layer():
    st.session_state.layer_idx = max(1, st.session_state.layer_idx - 1)
    _sync_sliders()

def inc_layer(n_total):
    st.session_state.layer_idx = min(n_total, st.session_state.layer_idx + 1)
    _sync_sliders()

def _sync_sliders():
    idx = st.session_state.layer_idx
    st.session_state.layer_idx_slider_design = idx
    st.session_state.layer_idx_slider_island = idx
    st.session_state.layer_idx_slider_assembly = idx

def _check_slider_sync():
    if st.session_state.get('force_slider_sync'):
        _sync_sliders()
        st.session_state.force_slider_sync = False

def render_design_view(final_geoms, settings, cur_idx, n_total, all_layer_geoms):
    _check_slider_sync()
    # Left: Tools (1), Right: Vis (3)
    c_tools, c_vis = st.columns([1, 3])
    
    with c_tools:
        st.markdown("##### Layer Control")
        col_prev, col_next = st.columns(2)
        col_prev.button("◀ Prev", use_container_width=True, key="prev_layer_btn_design", on_click=dec_layer)
        col_next.button("Next ▶", use_container_width=True, key="next_layer_btn_design", on_click=lambda: inc_layer(n_total))
        
        def on_slider_change():
            st.session_state.layer_idx = st.session_state.layer_idx_slider_design
            _sync_sliders()
        st.slider("Layer", 1, n_total, key="layer_idx_slider_design", on_change=on_slider_change, label_visibility="collapsed")
        
        st.divider()
        with st.expander("View Options", expanded=False):
            st.checkbox("Highlight Layer", key="show_slicer")
            st.checkbox("Show 3D Axes", key="show_3d_axes")
            show_shadowbox = st.checkbox("Show Shadowbox", value=False, key="design_show_shadowbox")
        
        st.caption("Snap View")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.button("Iso", on_click=set_cam, args=({'x':1.5,'y':-1.5,'z':1.5}, {'x':0,'y':0,'z':1}, 'design'), key="iso_btn", use_container_width=True)
        cc2.button("Top", on_click=set_cam, args=({'x':0,'y':0,'z':2.5}, {'x':0,'y':1,'z':0}, 'design'), key="top_btn", use_container_width=True)
        cc3.button("Side", on_click=set_cam, args=({'x':2.5,'y':0,'z':0}, {'x':0,'y':0,'z':1}, 'design'), key="side_btn", use_container_width=True)
        cc4.button("Front", on_click=set_cam, args=({'x':0,'y':2.5,'z':0}, {'x':0,'y':0,'z':1}, 'design'), key="front_btn", use_container_width=True)

    with c_vis:
        with st.spinner("Building 3D scene..."):
            fig = visualizer.create_3d_scene(final_geoms, settings, cur_idx, n_total, st.session_state.show_slicer, False, st.session_state.camera_snap_design, st.session_state.show_3d_axes, original_geoms=all_layer_geoms)
            fig.update_layout(height=800, margin=dict(l=0, r=0, t=0, b=0)) # Slightly shorter to fit
            if st.session_state.show_3d_axes:
                 fig.update_layout(scene=dict(zaxis=dict(range=[0, st.session_state.box_d], autorange=False)))
            fig.update_traces(selector=dict(type='mesh3d'), flatshading=False, lighting=dict(ambient=0.65, diffuse=0.5, specular=0.0, roughness=1.0, fresnel=0.0))
            
            # Add North Arrow
            arrow_x = st.session_state.box_w + (25 if st.session_state.get('design_show_shadowbox') else 10)
            fig.add_trace(go.Scatter3d(x=[arrow_x, arrow_x], y=[st.session_state.box_h-30, st.session_state.box_h-10], z=[0, 0], mode='lines+text', text=["N", ""], textfont=dict(color='#2196F3', size=20), line=dict(color='#2196F3', width=10), showlegend=False))
            fig.add_trace(go.Mesh3d(x=[arrow_x, arrow_x-2, arrow_x+2, arrow_x, arrow_x], y=[st.session_state.box_h, st.session_state.box_h-10, st.session_state.box_h-10, st.session_state.box_h-10, st.session_state.box_h-10], z=[0, 0, 0, 2, -2], i=[0, 0, 0, 0], j=[1, 2, 3, 4], k=[2, 3, 4, 1], color='#2196F3', flatshading=True))

            if st.session_state.get('design_show_shadowbox'):
                _add_shadowbox(fig, st.session_state.box_w, st.session_state.box_h, st.session_state.box_d)

        st.plotly_chart(fig, use_container_width=True, key="design_3d_view", config={'displaylogo': False})

def render_layer_tools(cur_idx, n_total, all_layer_geoms, final_geoms, layer_colors, bridge_status, raw_layer_geoms):
    """
    Renders the simplified Layer View with Auto-Heal visualization.
    Refactored to match Design/Assembly layout (Tools Left, Vis Right).
    """
    _check_slider_sync()
    
    # Main Layout: Tools Left (1), Vis Right (3)
    c_tools, c_vis = st.columns([1, 3])
    
    # --- TOOLS COLUMN ---
    with c_tools:
        st.markdown("##### Layer Control")
        
        # 1. Navigation
        col_prev, col_next = st.columns(2)
        key_prev = f"prev_layer_btn_island_L{cur_idx}"
        key_next = f"next_layer_btn_island_L{cur_idx}"
        
        col_prev.button("◀ Prev", disabled=(cur_idx <= 1), use_container_width=True, key=key_prev, on_click=dec_layer)
        col_next.button("Next ▶", disabled=(cur_idx >= n_total), use_container_width=True, key=key_next, on_click=lambda: inc_layer(n_total))
        
        def on_slider_change():
            st.session_state.layer_idx = st.session_state.layer_idx_slider_island
            _sync_sliders()
            
        st.slider("Layer", 1, n_total, key="layer_idx_slider_island", on_change=on_slider_change, label_visibility="collapsed")
        
        st.divider()
        
        # 2. Auto-Heal Stats
        mods = st.session_state.get('auto_modifications', [])
        current_layer_mods = mods[cur_idx - 1] if mods and cur_idx <= len(mods) else []
        
        # Calculate stats for current layer
        current_layer_polys = final_geoms[cur_idx - 1] if cur_idx <= len(final_geoms) else []
        num_auto_bridges = len([p for p in current_layer_polys if p.get('type') == 'auto_bridge'])
        
        if current_layer_mods:
            total_area = sum(m['area'] for m in current_layer_mods)
            st.success(f"**✅ Optimized {len(current_layer_mods)} spots**\n\n(+{total_area:.1f}mm² mass)")
        else:
            if num_auto_bridges > 0:
                st.info(f"Layer is robust.\n\n**🔗 {num_auto_bridges} Islands Connected**")
            else:
                st.info("No weak spots found.\nLayer is robust.")
        
        st.divider()

        # 3. View Options
        st.markdown("**View Options**")
        show_modifications = st.checkbox("Show Auto-Fixes", value=True, help="Green highlights show where mass was added.", key=f"show_mod_{cur_idx}")
        show_layer_below = st.checkbox("Show Layer Below", value=True, key=f"show_below_{cur_idx}")
        show_raw_diff = st.checkbox("Show Original Input", value=False, help="Show the raw geometry before healing (Red Dashed Line).", key=f"show_raw_{cur_idx}")
        
        st.caption("Structural Optimization is active.")

    # --- VISUALIZATION COLUMN ---
    with c_vis:
        # Determine effective mode (always view/island_selection)
        eff_mode = 'view'
        if st.session_state.get('tool_mode') == 'Islands':
             eff_mode = 'island_selection'
        
        # Prepare islands for UI if needed (for other tools)
        current_layer = all_layer_geoms[cur_idx-1]
        islands_for_ui = [p for p in current_layer if p['type'] == 'island']
        selected_indices = st.session_state.get('island_decisions', {}).get(f"L{cur_idx}", {}).get('selected', [])

        _render_2d_layer_view(cur_idx, final_geoms, all_layer_geoms, islands_for_ui, selected_indices, layer_colors, current_layer_mods, eff_mode, bridge_status, raw_layer_geoms, show_layer_below, show_modifications, show_raw_diff)

def render_assembly_tools(cur_idx, n_total, final_geoms, current_dowels):
    _check_slider_sync()
    layer_key = f"L{cur_idx}"
    current_layer_polys = final_geoms[cur_idx - 1]
    _handle_click_capture(f"assembly_plot_L{cur_idx}", f"last_click_assembly_L{cur_idx}", layer_key)
    
    # Layout: Tools Left, Vis Right
    c_tools, c_vis = st.columns([1, 3])
    
    with c_tools:
        st.markdown("##### Layer Control")
        col_prev, col_next = st.columns(2)
        col_prev.button("◀ Prev", use_container_width=True, key="prev_layer_btn_assembly", on_click=dec_layer)
        col_next.button("Next ▶", use_container_width=True, key="next_layer_btn_assembly", on_click=lambda: inc_layer(n_total))
        
        def on_slider_change():
            st.session_state.layer_idx = st.session_state.layer_idx_slider_assembly
            _sync_sliders()
        st.slider("Layer", 1, n_total, key="layer_idx_slider_assembly", on_change=on_slider_change, label_visibility="collapsed")
        st.divider()

        # Simplified Settings (No Manual/History tabs)
        st.info("Mode: Grid Supports (Auto-Generated).")
        st.checkbox("Fluid Smoothing", value=True, key="jig_fluid", help="Applies organic fillets to intersections for a 'molded' look.")
    
    with c_vis:
        # Render Assembly
        # Pass empty list for jig_mods since we removed the ability to create them
        _render_2d_assembly_view(cur_idx, current_layer_polys, current_dowels, [], show_ids=False, highlight_selected=False)


def render_preview(final_geoms, settings, n_total, all_layer_geoms):
    # Layout: Tools Left, Vis Right
    c_tools, c_vis = st.columns([1, 3])
    
    with c_tools:
        st.markdown("##### Preview Controls")
        st.caption("Snap View")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.button("Iso", on_click=set_cam, args=({'x':1.5,'y':-1.5,'z':1.5}, {'x':0,'y':0,'z':1}, 'preview'), key="iso_btn_prev", use_container_width=True)
        cc2.button("Top", on_click=set_cam, args=({'x':0,'y':0,'z':2.5}, {'x':0,'y':1,'z':0}, 'preview'), key="top_btn_prev", use_container_width=True)
        cc3.button("Side", on_click=set_cam, args=({'x':2.5,'y':0,'z':0}, {'x':0,'y':0,'z':1}, 'preview'), key="side_btn_prev", use_container_width=True)
        cc4.button("Front", on_click=set_cam, args=({'x':0,'y':2.5,'z':0}, {'x':0,'y':0,'z':1}, 'preview'), key="front_btn_prev", use_container_width=True)
        
        st.divider()
        color_presets = {"Birch": "#E3C099", "Oak": "#D2B48C", "Pine": "#E5C6A0", "Walnut": "#5D4037", "Mahogany": "#4A0404", "White": "#FFFFFF", "Grey": "#808080", "Dark Grey": "#333333", "Black": "#000000"}
        wood_choice = st.selectbox("Wood Color", list(color_presets.keys()), index=0, key="wood_color_select")
        wood_color = color_presets[wood_choice]
        show_shadowbox = st.checkbox("Show Shadowbox", value=False)

    with c_vis:
        with st.spinner("Building full 3D preview..."):
            fig = visualizer.create_3d_scene(final_geoms, settings, n_total, n_total, False, False, st.session_state.camera_snap_preview, st.session_state.show_3d_axes, original_geoms=all_layer_geoms, color_mode='wood')
            fig.update_layout(height=900, margin=dict(l=0, r=0, t=0, b=0))
            if st.session_state.show_3d_axes:
                fig.update_layout(scene=dict(zaxis=dict(range=[0, st.session_state.box_d], autorange=False)))
            
            # Apply wood color
            mesh_traces = [t for t in fig.data if t.type == 'mesh3d']
            h_base = wood_color.lstrip('#')
            rgb_base = tuple(int(h_base[j:j+2], 16) for j in (0, 2, 4))
            for i, trace in enumerate(mesh_traces):
                factor = 0.75 + (0.35 * i / max(1, len(mesh_traces) - 1))
                new_rgb = [min(255, max(0, int(c * factor))) for c in rgb_base]
                trace.color = f"#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}"
            
            # Apply edge color
            edge_color = '#666666' if wood_color.lower() in ['#000000', 'black'] else ('#999999' if wood_color.lower() in ['#ffffff', 'white'] else f"#{max(0, int(rgb_base[0]*0.6)):02x}{max(0, int(rgb_base[1]*0.6)):02x}{max(0, int(rgb_base[2]*0.6)):02x}")
            for trace in fig.data:
                if isinstance(trace, go.Scatter3d) and trace.mode == 'lines' and trace.line.color != '#2196F3':
                    trace.line.color = edge_color

            if show_shadowbox:
                _add_shadowbox(fig, st.session_state.box_w, st.session_state.box_h, st.session_state.box_d, color="#333333")

        st.plotly_chart(fig, use_container_width=True, key="preview_3d_view", config={'displaylogo': False})

def render_export(final_geoms, settings, current_dowels):
    wood_choice = st.session_state.get("wood_color_select", "Birch")
    color_presets = {"Birch": "#E3C099", "Oak": "#D2B48C", "Pine": "#E5C6A0", "Walnut": "#5D4037", "Mahogany": "#4A0404", "White": "#FFFFFF", "Grey": "#808080", "Dark Grey": "#333333", "Black": "#000000"}
    ex_wood_color = color_presets.get(wood_choice, "#E3C099")
    
    c_ex1, c_ex2 = st.columns([3, 1])
    with c_ex2:
        st.markdown("##### Export Settings")
        export_fmt = st.selectbox("Format", ["SVG", "DXF", "PDF", "PNG", "JPG"], index=1)
        
        st.download_button("📄 Download Assembly Guide (PDF)", 
                           data=exporter.generate_assembly_guide_pdf(final_geoms, settings, st.session_state.jig_modifications, current_dowels),
                           file_name=f"{st.session_state.proj_name}_AssemblyGuide.pdf", mime="application/pdf", use_container_width=True)
        
        st.download_button(f"⬇️ Download {export_fmt} ZIP", 
                           data=exporter.generate_zip_data(final_geoms, settings, st.session_state.jig_modifications, current_dowels, export_fmt),
                           file_name=f"{st.session_state.proj_name}_{export_fmt}.zip", mime="application/zip", type="primary", use_container_width=True)
        
        st.divider()
        st.markdown("##### Nesting")
        nest_mode = st.radio("Sheet Strategy", ["Fixed Size", "Auto-Size"], index=0, horizontal=True)
        
        ex_sheet_w, ex_sheet_h = 600, 400
        if nest_mode == "Fixed Size":
            c_n1, c_n2 = st.columns(2)
            ex_sheet_w = c_n1.number_input("Sheet W (mm)", value=600, step=10)
            ex_sheet_h = c_n2.number_input("Sheet H (mm)", value=400, step=10)
            nest_aspect = ex_sheet_w / ex_sheet_h
        else:
            nest_aspect = st.number_input("Target Aspect (L/W)", min_value=1.0, value=1.5, step=0.1)

        nest_gap = st.number_input("Min Gap (mm)", value=4.0, min_value=0.0, step=0.5)
        nest_rotation = st.checkbox("Allow Rotation", value=True)
        
        if st.button("🧩 Calculate Layout", type="primary", use_container_width=True):
            _run_nesting(final_geoms, current_dowels, nest_gap, nest_rotation, nest_aspect, nest_mode, ex_sheet_w, ex_sheet_h)
            st.rerun()
            
        if 'nested_components' in st.session_state:
            n_sheets_c = len(st.session_state.nested_components)
            n_sheets_j = len(st.session_state.get('nested_jigs', []))
            st.success(f"Packed into {n_sheets_c} Component Sheet(s) + {n_sheets_j} Jig Sheet(s)")
            
            # Cache the expensive ZIP generation.
            # Fix: Use '_' prefix to prevent Streamlit from trying to hash complex geometry objects.
            # We pass 'nesting_timestamp' as the cache key to force updates when data changes.
            @st.cache_data
            def get_cached_zip(_nc, _nj, _ncd, _njd, fmt, wc, ec, pname, version_id):
                return exporter.generate_nested_zip(_nc, _nj, _ncd, _njd, fmt, wc, ec)
            
            nest_ver = st.session_state.get('nesting_version', 0)
            zip_data = get_cached_zip(st.session_state.nested_components, st.session_state.get('nested_jigs'), 
                                     st.session_state.nested_comp_dims, st.session_state.get('nested_jig_dims'), 
                                     export_fmt, ex_wood_color, "#000000", st.session_state.proj_name, nest_ver)

            st.download_button(f"⬇️ Download Nested {export_fmt} ZIP", 
                               data=zip_data,
                               file_name=f"{st.session_state.proj_name}_Nested_{export_fmt}.zip", mime="application/zip", use_container_width=True)

    with c_ex1:
        st.markdown("#### Export Preview")
        
        tab_layers, tab_nesting = st.tabs(["Individual Layers", "Nested Sheets"])
        
        with tab_layers:
            for i, layer_polys in enumerate(final_geoms):
                layer_num = i + 1
                with st.container():
                    st.markdown(f"**Layer {layer_num}**")
                    c_part, c_jig = st.columns(2)
                    svg_parts = geometry_engine.generate_svg_string(layer_polys, st.session_state.box_w, st.session_state.box_h, fill_color=ex_wood_color, stroke_color="black", add_background=False)
                    b64_parts = base64.b64encode(svg_parts.encode('utf-8')).decode("utf-8")
                    c_part.markdown(f'<img src="data:image/svg+xml;base64,{b64_parts}" style="width: 100%;">', unsafe_allow_html=True)
                    
                    jig_mods = st.session_state.jig_modifications.get(f"L{layer_num}", [])
                    jig_data = geometry_engine.generate_jig_geometry(layer_polys, current_dowels, st.session_state.box_w, st.session_state.box_h, layer_num, jig_mods, conn_width=st.session_state.jig_conn_width, grid_spacing=st.session_state.get('jig_grid_spacing', 20.0), fluid_smoothing=st.session_state.get('jig_fluid', True))
                    if jig_data:
                        svg_jig = geometry_engine.generate_svg_string([{'poly': jig_data['poly']}], st.session_state.box_w, st.session_state.box_h, fill_color="#de2d26", stroke_color="#a50f15", add_background=False, fill_opacity=0.8)
                        b64_jig = base64.b64encode(svg_jig.encode('utf-8')).decode("utf-8")
                        c_jig.markdown(f'<img src="data:image/svg+xml;base64,{b64_jig}" style="width: 100%;">', unsafe_allow_html=True)
                    else: c_jig.info("Not required")
                st.divider()
        
        with tab_nesting:
            if 'nested_components' in st.session_state:
                st.info(f"Sheet Size: {st.session_state.nested_comp_dims[0]:.0f} x {st.session_state.nested_comp_dims[1]:.0f} mm")
                
                # Render Components Sheets
                comps = st.session_state.nested_components
                for i, sheet in enumerate(comps):
                    st.markdown(f"**Component Sheet {i+1}**")
                    # Construct poly list for SVG
                    sheet_polys = [{'poly': item['poly']} for item in sheet]
                    svg_s = geometry_engine.generate_svg_string(sheet_polys, st.session_state.nested_comp_dims[0], st.session_state.nested_comp_dims[1], fill_color=ex_wood_color, stroke_color="black", add_background=True)
                    b64_s = base64.b64encode(svg_s.encode('utf-8')).decode("utf-8")
                    st.markdown(f'<img src="data:image/svg+xml;base64,{b64_s}" style="width: 100%;">', unsafe_allow_html=True)
                    st.divider()
                
                # Render Jig Sheets (if any)
                jigs = st.session_state.get('nested_jigs', [])
                for i, sheet in enumerate(jigs):
                    st.markdown(f"**Jig Sheet {i+1}**")
                    sheet_polys = [{'poly': item['poly']} for item in sheet]
                    svg_j = geometry_engine.generate_svg_string(sheet_polys, st.session_state.nested_jig_dims[0], st.session_state.nested_jig_dims[1], fill_color="#de2d26", stroke_color="#a50f15", add_background=True, fill_opacity=0.8)
                    b64_j = base64.b64encode(svg_j.encode('utf-8')).decode("utf-8")
                    st.markdown(f'<img src="data:image/svg+xml;base64,{b64_j}" style="width: 100%;">', unsafe_allow_html=True)
                    st.divider()

            else:
                st.info("Click 'Calculate Layout' to see the nesting preview.")

# --- Helpers ---

def _handle_click_capture(plot_key, last_click_key, layer_key):
    chart_state = st.session_state.get(plot_key)
    new_px, new_py, new_pw, new_ph = None, None, None, None
    if chart_state and "selection" in chart_state:
        selection = chart_state["selection"]
        if "box" in selection and selection["box"]:
            box = selection["box"][0]
            x_range, y_range = box.get("x", []), box.get("y", [])
            if x_range and y_range:
                new_px, new_py = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
                new_pw, new_ph = abs(x_range[1] - x_range[0]), abs(y_range[1] - y_range[0])
        elif "points" in selection and selection["points"]:
            pt = selection["points"][0]
            new_px, new_py = float(pt['x']), float(pt['y'])
    
    if new_px is not None:
        current = (new_px, new_py, new_pw, new_ph)
        if st.session_state.get(last_click_key) != current:
            st.session_state[f"jig_x_{layer_key}"] = new_px
            st.session_state[f"jig_y_{layer_key}"] = new_py
            if new_pw: st.session_state[f"jig_w_{layer_key}"] = new_pw
            if new_ph: st.session_state[f"jig_h_{layer_key}"] = new_ph
            st.session_state[last_click_key] = current
            st.toast(f"Captured: {new_px:.1f}, {new_py:.1f}", icon="📍")

def _add_shadowbox(fig, w, h, d, color='#333333'):
    th = 15.0
    def make_box(x0, x1, y0, y1, z0, z1):
        return go.Mesh3d(x=[x0, x1, x1, x0, x0, x1, x1, x0], y=[y0, y0, y1, y1, y0, y0, y1, y1], z=[z0, z0, z0, z0, z1, z1, z1, z1],
                         i=[0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1], j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6], k=[2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5],
                         color=color, opacity=1.0, flatshading=True)
    for args in [(-th, w+th, -th, h+th, -th, 0), (-th, 0, -th, h+th, 0, d), (w, w+th, -th, h+th, 0, d), (0, w, -th, 0, 0, d), (0, w, h, h+th, 0, d)]:
        fig.add_trace(make_box(*args))

def _render_history_items(layer_key, cur_idx, raw_layer_geoms=None):
    def set_del(type_, idx): st.session_state.item_to_delete = {'type': type_, 'layer': layer_key, 'index': idx}
    
    highlighted_item = st.session_state.get('history_highlight', None)
    
    def set_highlight(val): st.session_state.history_highlight = val

    for i, group in enumerate(st.session_state.merge_groups.get(layer_key, [])):
        c1, c2 = st.columns([1, 4])
        with c1: st.button("🔍", key=f"show_merge_{cur_idx}_{i}", on_click=set_highlight, args=({'type': 'merge', 'data': group},), use_container_width=True)
        with c2: st.button(f"🗑️ Merge: {group}", key=f"del_merge_{cur_idx}_{i}", on_click=set_del, args=('merge', i), use_container_width=True)
    
    for i, idx in enumerate(st.session_state.bridge_to_mainland_requests.get(layer_key, [])):
        c1, c2 = st.columns([1, 4])
        with c1: st.button("🔍", key=f"show_bridge_{cur_idx}_{i}", on_click=set_highlight, args=({'type': 'bridge', 'data': idx},), use_container_width=True)
        with c2: st.button(f"🗑️ Bridge Main: #{idx+1}", key=f"del_bridge_{cur_idx}_{i}", on_click=set_del, args=('bridge', i), use_container_width=True)
    
    for i, mp in enumerate(st.session_state.manual_problems.get(layer_key, [])):
        c1, c2 = st.columns([1, 4])
        with c1: st.button("🔍", key=f"show_man_prob_{cur_idx}_{i}", on_click=set_highlight, args=({'type': 'manual_prob', 'data': mp},), use_container_width=True)
        with c2: st.button(f"🗑️ Flag #{i+1}", key=f"del_man_prob_{cur_idx}_{i}", on_click=set_del, args=('manual_prob', i), use_container_width=True)
    
    # Add Problem Decisions
    layer_probs = [k for k,v in st.session_state.problem_decisions.items() if k.startswith(f"{layer_key}_") and v.get('action') != 'remove']
    for i, p_key in enumerate(layer_probs):
        action = st.session_state.problem_decisions[p_key].get('action', 'remove')
        idx_str = p_key.split('_')[-1]
        def reset_prob(pk): 
            del st.session_state.problem_decisions[pk]
            st.rerun()
        
        c1, c2 = st.columns([1, 4])
        # Pass raw_layer_geoms via closure/args not possible easily to button args? 
        # Actually we pass 'data' as p_key. The resolution loop needs raw_layer_geoms.
        with c1: st.button("🔍", key=f"show_prob_dec_{p_key}", on_click=set_highlight, args=({'type': 'prob_dec', 'data': p_key},), use_container_width=True)
        with c2: st.button(f"↩️ {action.title()} {idx_str}", key=f"undo_prob_{p_key}", on_click=reset_prob, args=(p_key,), use_container_width=True)

def _render_2d_layer_view(cur_idx, final_geoms, all_layer_geoms, islands_for_ui, selected_indices, layer_colors, modifications, tool_mode, bridge_status, raw_layer_geoms=None, show_layer_below=True, show_modifications=True, show_raw_diff=False):
    layer_key = f"L{cur_idx}"
    current_layer_polys = [p for p in final_geoms[cur_idx - 1] if p['type'] != 'problem']
    visible_problems = [p for p in final_geoms[cur_idx - 1] if p['type'] == 'problem']
    
    polys_to_highlight = [islands_for_ui[i] for i in selected_indices if i < len(islands_for_ui)]
    manual_points = st.session_state.manual_bridge_points.get(layer_key, [])
    original_geoms = all_layer_geoms[cur_idx-1] if st.session_state.show_original_islands else None
    
    # Resolve History Highlight
    history_highlight_polys = []
    hh = st.session_state.get('history_highlight')
    if hh:
        try:
            if hh['type'] == 'merge':
                for idx in hh['data']:
                    if idx < len(islands_for_ui): history_highlight_polys.append(islands_for_ui[idx]['poly'])
            elif hh['type'] == 'bridge':
                idx = hh['data']
                if idx < len(islands_for_ui): history_highlight_polys.append(islands_for_ui[idx]['poly'])
            elif hh['type'] == 'manual_prob':
                mp = hh['data']
                history_highlight_polys.append(box(mp['x'] - mp['w']/2, mp['y'] - mp['h']/2, mp['x'] + mp['w']/2, mp['y'] + mp['h']/2))
            elif hh['type'] == 'prob_dec':
                # Use raw_layer_geoms if available to find the original problem
                p_key = hh['data']
                # Key format: L{layer}_P_{id} OR L{layer}_P{int} (legacy) OR L{layer}_M{int}
                
                parts = p_key.split('_')
                # parts[0] = L1
                # parts[1] = P or M?
                # New format: L1_P_X123_Y456 -> parts: [L1, P, X123, Y456]? 
                # Wait, split('_') splits everything.
                
                if len(parts) >= 2 and raw_layer_geoms:
                    if '_M' in p_key: # Manual
                        # Extract int index from last part? L1_M5 -> M5
                        try:
                            m_part = [p for p in parts if p.startswith('M')][0]
                            idx = int(m_part[1:])
                            manual_boxes = st.session_state.manual_problems.get(layer_key, [])
                            if idx < len(manual_boxes):
                                mp = manual_boxes[idx]
                                history_highlight_polys.append(box(mp['x'] - mp['w']/2, mp['y'] - mp['h']/2, mp['x'] + mp['w']/2, mp['y'] + mp['h']/2))
                        except: pass
                    else:
                        # Auto Problem
                        # Try to find ID in raw layer
                        # ID is everything after 'L{layer}_P_'
                        # e.g. "L1_P_X100_Y200"
                        if '_P_' in p_key:
                             target_id = p_key.split('_P_')[1] # X100_Y200
                        else:
                             # Legacy integer Fallback?
                             target_id = parts[-1] 
                             if target_id.startswith('P'): target_id = target_id[1:] # handle P0 -> 0

                        raw_layer = raw_layer_geoms[cur_idx-1]
                        # Search by ID
                        found = [p['poly'] for p in raw_layer if str(p.get('orig_idx', '')) == str(target_id) and p['type'] == 'problem']
                        if found: history_highlight_polys.extend(found)

        except Exception as e:
            print(f"Highlight resolve error: {e}")

    # Prepare Raw Diff Polygon
    diff_poly_for_viz = None
    if show_raw_diff and raw_layer_geoms:
        try:
             # Union all polys in the raw layer (islands, safe, terrain)
             r_layer = raw_layer_geoms[cur_idx-1]
             # Filter empty or problems? Usually raw includes everything.
             # We just want to see the shape.
             valid_raw = [p['poly'] for p in r_layer if not p['poly'].is_empty and p['type'] in ['island', 'safe', 'terrain', 'problem']]
             
             # DEBUG: Verify we found data
             # st.warning(f"DEBUG: Found {len(valid_raw)} raw polygons on this layer.")
             
             if valid_raw:
                 diff_poly_for_viz = unary_union(valid_raw)
                 
                 # DIAGNOSTIC: Show Area Delta & Vertex Count
                 raw_area = diff_poly_for_viz.area
                 final_area = unary_union([p['poly'] for p in current_layer_polys]).area
                 delta = final_area - raw_area
                 
                 # Count Vertices
                 n_verts = 0
                 if diff_poly_for_viz.geom_type == 'MultiPolygon':
                     for g in diff_poly_for_viz.geoms: n_verts += len(g.exterior.coords)
                 else:
                     n_verts = len(diff_poly_for_viz.exterior.coords)
                     
                 cleanup_status = "CLEANED (Rounded)" if st.session_state.get('auto_cleanup', True) else "RAW (Unprocessed)"
                 
                 st.caption(f"**Diagnostic:** Area Δ: {delta:+.2f}mm² | Vertices: {n_verts}")
                 st.caption(f"**Source Status:** {cleanup_status} (Toggle in Generation Settings -> Intelligent Automation)")
             else:
                 st.error("No valid raw geometry found for this layer!")
                 
        except Exception as e: st.error(f"Raw Diff Error: {e}")

    fig_2d = visualizer.create_2d_view(
        polygons=current_layer_polys, width_mm=st.session_state.box_w, height_mm=st.session_state.box_h,
        polygons_to_highlight=polys_to_highlight, original_polys=original_geoms, manual_points=manual_points,
        layer_color=layer_colors[cur_idx - 1], problem_polys=visible_problems if tool_mode == "Problems" else None,
        active_problem_poly=None, layer_index=cur_idx,
        modifications_polys=modifications if show_modifications else None,
        history_highlight_polys=history_highlight_polys,
        raw_diff_geometry=diff_poly_for_viz
    )
    
    # Post-processing: Remove default annotations, add box border
    fig_2d.layout.annotations = []
    fig_2d.layout.shapes = [] # Remove default North Arrow shapes from visualizer
    fig_2d.add_shape(type="rect", x0=0, y0=0, x1=st.session_state.box_w, y1=st.session_state.box_h, line=dict(color="black", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
    
    # Robust "Layer Below" Rendering: Draw manually on top so it never gets hidden by opaque layers
    if show_layer_below and cur_idx > 1:
        prev_layer = final_geoms[cur_idx - 2]
        # st.write(f"DEBUG: Layer Below Active. Prev Layer Size: {len(prev_layer)}") # Visible Debug
        
        count = 0
        for p in prev_layer:
            if not p['poly'].is_empty:
                geoms = list(p['poly'].geoms) if p['poly'].geom_type == 'MultiPolygon' else [p['poly']]
                for g in geoms:
                    x, y = g.exterior.xy
                    fig_2d.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='grey', width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
                    count += 1
        
        if count == 0:
             if len(prev_layer) > 0:
                 st.toast(f"Below Layer has {len(prev_layer)} polys but 0 drawn!", icon="⚠️")
             else:
                 pass
        else:
             pass

    if manual_points:
        statuses = bridge_status.get(layer_key, {})
        x_s, y_s, t_s = [], [], []
        x_f, y_f, t_f = [], [], []
        for i, pt in enumerate(manual_points):
            if statuses.get(i) == "Success": x_s.append(pt['x']); y_s.append(pt['y']); t_s.append(str(i+1))
            else: x_f.append(pt['x']); y_f.append(pt['y']); t_f.append(str(i+1))
        if x_s: fig_2d.add_trace(go.Scatter(x=x_s, y=y_s, mode='markers+text', marker=dict(color='green', size=10), text=t_s, textposition="top center", name="Valid"))
        if x_f: fig_2d.add_trace(go.Scatter(x=x_f, y=y_f, mode='markers+text', marker=dict(color='red', size=10, symbol='x'), text=t_f, textposition="top center", name="Invalid"))

    # Restore Blue North Arrow (Outside Plot)
    arrow_len = 30
    head_len = 10
    arrow_x = st.session_state.box_w + 10
    arrow_y_tip = st.session_state.box_h
    arrow_y_base = arrow_y_tip - arrow_len
    arrow_y_head_base = arrow_y_tip - head_len

    fig_2d.add_trace(go.Scatter(
        x=[arrow_x, arrow_x], y=[arrow_y_base, arrow_y_head_base],
        mode='lines+text', text=["N", ""], textfont=dict(color='#2196F3', size=20), textposition="bottom center",
        line=dict(color='#2196F3', width=4), showlegend=False, hoverinfo='none'
    ))
    fig_2d.add_trace(go.Scatter(
        x=[arrow_x, arrow_x-2, arrow_x+2, arrow_x],
        y=[arrow_y_tip, arrow_y_head_base, arrow_y_head_base, arrow_y_tip],
        fill='toself', mode='none', fillcolor='#2196F3', showlegend=False, hoverinfo='none'
    ))

    fig_2d.update_layout(dragmode='select', clickmode='event+select')
    st.plotly_chart(fig_2d, use_container_width=True, key="island_2d_view_persistent", config={'displayModeBar': True, 'displaylogo': False}, on_select="rerun", selection_mode=["points", "box"])

def _render_2d_assembly_view(cur_idx, current_layer_polys, current_dowels, jig_mods, show_ids=False, highlight_selected=True):
    layer_key = f"L{cur_idx}"
    eff_dowels = current_dowels if st.session_state.get("show_dowels_ui", True) else []
    
    fig_2d = visualizer.create_2d_view(polygons=current_layer_polys, width_mm=st.session_state.box_w, height_mm=st.session_state.box_h, layer_index=cur_idx)
    fig_2d.layout.annotations = []
    fig_2d.layout.shapes = []
    fig_2d.add_shape(type="rect", x0=0, y0=0, x1=st.session_state.box_w, y1=st.session_state.box_h, line=dict(color="black", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

    # Highlight Selected Islands (Sync with Layer Tools selection)
    if highlight_selected:
        selected_indices = st.session_state.get('island_decisions', {}).get(f"L{cur_idx}", {}).get('selected', [])
        # We need to map these indices to polys? 
        # current_layer_polys contains islands.
        # We can just iterate and match index?
        # Islands in current_layer_polys typically preserve order.
        parts = [p for p in current_layer_polys if 'poly' in p and not p['poly'].is_empty]
        for i, p in enumerate(parts):
            if i in selected_indices:
                # Highlight
                poly = p['poly']
                if poly.is_empty: continue
                x, y = poly.exterior.xy
                fig_2d.add_trace(go.Scatter(x=list(x), y=list(y), fill="toself", mode='lines', line=dict(color='orange', width=2), fillcolor='rgba(255, 165, 0, 0.3)', name=f"Sel #{i+1}"))
    
    # Show Part IDs
    if show_ids:
        parts = [p for p in current_layer_polys if 'poly' in p and not p['poly'].is_empty]
        for i, p in enumerate(parts):
            poly = p['poly']
            if poly.is_empty: continue
            c = poly.centroid
            fig_2d.add_trace(go.Scatter(
                x=[c.x], y=[c.y],
                mode='text',
                text=[str(i+1)],
                textfont=dict(color='black', size=12),
                showlegend=False
            ))

    # Use eff_dowels for jig generation so jig respects visibility? 
    # Actually, jig usually *needs* dowels to form holes. 
    # But if user hides dowels, maybe they want to see jig without holes?
    # Or maybe this just affects the red dowel rendering if that's separate?
    # generate_jig_geometry cuts holes for dowels. Let's keep that logic but maybe visualizer handles the red spots.
    # visualizer.create_2d_view DOES NOT plot dowels automatically unless we pass them or something?
    # Re-reading create_2d_view... it doesn't seem to take dowels as main arg.
    # Dowels are usually baked into layer geometry? No, they are separate in data structure.
    # Let's see... create_2d_view signature: polygons, ... manual_points ... problem_polys...
    # It doesn't seem to paint dowels directly.
    # render_layer_tools passes 'final_geoms' which might have holes?
    # But for assembly view, we want to see the JIG and the alignment.
    
    jig_data = geometry_engine.generate_jig_geometry(current_layer_polys, eff_dowels, st.session_state.box_w, st.session_state.box_h, cur_idx, jig_mods, conn_width=st.session_state.jig_conn_width, grid_spacing=st.session_state.get('jig_grid_spacing', 20.0), fluid_smoothing=st.session_state.get('jig_fluid', True))
    if jig_data:
        jig_geom = jig_data['poly']
        polys_to_plot = []
        if jig_geom.geom_type == 'Polygon': polys_to_plot = [jig_geom]
        elif jig_geom.geom_type == 'MultiPolygon': polys_to_plot = list(jig_geom.geoms)
        
        for poly in polys_to_plot:
            # Correctly handle holes by separating rings with None
            x_combined = list(poly.exterior.coords.xy[0]) + [None]
            y_combined = list(poly.exterior.coords.xy[1]) + [None]
            for interior in poly.interiors:
                x_combined.extend(list(interior.coords.xy[0]) + [None])
                y_combined.extend(list(interior.coords.xy[1]) + [None])
                
            fig_2d.add_trace(go.Scatter(
                x=x_combined, y=y_combined, fill="toself",
                mode='lines', line=dict(color='#a50f15', width=1),
                fillcolor='rgba(222, 45, 38, 0.8)',
                name="Jig"
            ))

    if jig_mods:
        sel_idx = st.session_state.get(f"jig_sel_{cur_idx}")
        for i, mod in enumerate(jig_mods):
            color = 'green' if mod['type'] == 'add' else 'red'
            opacity = 0.3 if i == sel_idx else 0.1
            fig_2d.add_shape(type="rect", x0=mod['x']-mod['w']/2, y0=mod['y']-mod['h']/2, x1=mod['x']+mod['w']/2, y1=mod['y']+mod['h']/2, line=dict(color=color), fillcolor=color, opacity=opacity)

    # Restore Blue North Arrow (Outside Plot)
    arrow_len = 30
    head_len = 10
    arrow_x = st.session_state.box_w + 10
    arrow_y_tip = st.session_state.box_h
    arrow_y_base = arrow_y_tip - arrow_len
    arrow_y_head_base = arrow_y_tip - head_len

    fig_2d.add_trace(go.Scatter(
        x=[arrow_x, arrow_x], y=[arrow_y_base, arrow_y_head_base],
        mode='lines+text', text=["N", ""], textfont=dict(color='#2196F3', size=20), textposition="bottom center",
        line=dict(color='#2196F3', width=4), showlegend=False, hoverinfo='none'
    ))
    fig_2d.add_trace(go.Scatter(
        x=[arrow_x, arrow_x-2, arrow_x+2, arrow_x],
        y=[arrow_y_tip, arrow_y_head_base, arrow_y_head_base, arrow_y_tip],
        fill='toself', mode='none', fillcolor='#2196F3', showlegend=False, hoverinfo='none'
    ))

    fig_2d.update_layout(dragmode='select', clickmode='event+select')
    st.plotly_chart(fig_2d, use_container_width=True, key=f"assembly_plot_L{cur_idx}", config={'displayModeBar': True, 'displaylogo': False}, on_select="rerun", selection_mode=["points", "box"])

def _run_nesting(final_geoms, current_dowels, gap, rotation, aspect, mode, fw, fh):
    with st.spinner("Calculating optimal layout..."):
        all_components = []
        all_jigs = []
        for i, layer in enumerate(final_geoms):
            for p in layer:
                if 'poly' in p and not p['poly'].is_empty: all_components.append({'poly': p['poly'], 'layer': i+1, 'type': 'part'})
            jig_mods = st.session_state.jig_modifications.get(f"L{i+1}", [])
            jig_data = geometry_engine.generate_jig_geometry(layer, current_dowels, st.session_state.box_w, st.session_state.box_h, i+1, jig_mods, conn_width=st.session_state.jig_conn_width, grid_spacing=st.session_state.get('jig_grid_spacing', 20.0), fluid_smoothing=st.session_state.get('jig_fluid', True))
            if jig_data: all_jigs.append({'poly': jig_data['poly'], 'layer': i+1, 'type': 'jig'})

        def pack(items):
            if not items: return [], 0, 0
            
            # Helper to run pack
            def run_packer(w, h):
                packer = nesting.MultiSheetPacker(w, h, gap, allow_rotation=rotation, grid_step=max(w,h)/150)
                packer.pack_items(items)
                return packer.sheets

            if mode == "Fixed Size":
                return run_packer(fw, fh), fw, fh
            else:
                # Auto-Size (Infinite Sheet Heuristic)
                total_area = sum(i['poly'].area for i in items)
                est_w = math.sqrt(total_area / 0.8 * aspect)
                est_h = est_w / aspect
                # Try iteratively
                for i in range(20):
                    w, h = est_w * (1 + 0.05*i), est_h * (1 + 0.05*i)
                    sheets = run_packer(w, h)
                    if len(sheets) == 1:
                        return sheets, w, h
                return sheets, w, h # Return whatever we got (likely last attempt)

        comps, cw, ch = pack(all_components)
        jigs, jw, jh = pack(all_jigs)
        import time
        st.session_state.nested_components = comps
        st.session_state.nested_comp_dims = (cw, ch)
        st.session_state.nested_jigs = jigs
        st.session_state.nested_jig_dims = (jw, jh)
        # Increment Cache Version
        st.session_state.nesting_version = time.time()
        st.toast("Nesting complete!", icon="🧩")