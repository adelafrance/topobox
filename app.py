# app.py

import streamlit as st
import numpy as np
import math
import copy
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Streamlit
from plotly.colors import sample_colorscale, get_colorscale

from utils import data_loader, geometry_engine, project_manager, app_state, app_layout, views

st.set_page_config(page_title="TopoBox Pro", layout="wide", initial_sidebar_state="collapsed")
app_layout.inject_custom_css()

# --- Definitive Warning Suppression ---
# The "on_click" bug in st.plotly_chart triggers a Streamlit UI warning that bypasses
# standard Python warning filters. The only way to suppress this cosmetic bug is to
# intercept the st.warning function itself and filter out the erroneous message.
_original_warning = st.warning
def _suppressed_warning(body, *args, **kwargs):
    if isinstance(body, str) and "The keyword arguments have been deprecated" in body:
        return  # Suppress this specific warning by doing nothing
    _original_warning(body, *args, **kwargs)
st.warning = _suppressed_warning

API_KEY = data_loader.load_api_key()

# --- STATE INITIALIZATION ---
app_state.initialize_session_state()
if 'user_mode' not in st.session_state: st.session_state.user_mode = 'creator'
if 'admin_pwd' not in st.session_state: st.session_state.admin_pwd = ""

app_state.restore_autosave()
app_state.run_migrations()
app_state.save_autosave()
app_state.handle_deferred_deletion()

# --- SECURITY REDIRECT ---
# If Creator is on a Maker-only view, redirect to Preview
if st.session_state.user_mode == 'creator' and st.session_state.current_view in ["Layer Tools", "Assembly Tools", "Export"]:
    st.session_state.current_view = "Preview"

# --- CALLBACKS & HELPERS ---
def process_data_callback():
    st.session_state.run_btn = True
    st.session_state.is_new_run = True
    st.session_state.show_adjustment_info = False

# --- UI ---
app_layout.render_navigation()
app_layout.render_sidebar(API_KEY, process_data_callback)

# --- MAIN LOGIC ---
if not st.session_state.run_btn and st.session_state.elevation_data is None:
    st.info("Please configure your settings and click 'Load/Process Data'.")

if st.session_state.run_btn or st.session_state.elevation_data is not None:
    # Version Control for Cache Invalidation
    # We bump this when we change geometry logic to ensure users don't see stale cached data.
    CURRENT_CODE_VERSION = "v3.9_DebrisFilter"
    
    # --- Auto-update on Location Change ---
    # If location parameters change, we must force a re-fetch of the data.
    loc_keys = ['lat', 'lon', 'width_km', 'height_km']
    loc_changed = False
    for k in loc_keys:
        if f'last_{k}' not in st.session_state: st.session_state[f'last_{k}'] = st.session_state[k]
        if st.session_state[k] != st.session_state[f'last_{k}']:
            loc_changed = True
            st.session_state[f'last_{k}'] = st.session_state[k]
    
    if loc_changed:
        st.session_state.elevation_data = None
        # Clear modifications as the terrain is totally different
        for k in ['island_decisions', 'merge_groups', 'merge_group_names', 'bridge_to_mainland_requests', 'manual_bridge_points', 'jig_modifications', 'problem_decisions', 'manual_problems']: st.session_state[k] = {}
        # Reset critical params baseline to prevent warning on this pass
        for p in ['blur', 'min_area', 'box_w', 'box_h']: st.session_state[f'last_{p}'] = st.session_state[p]

    # --- Intelligent State Persistence ---
    # Detect if critical parameters changed
    critical_params = ['blur', 'min_area', 'min_feature_width', 'box_w', 'box_h', 'auto_cleanup', 'auto_bridge', 'auto_fuse', 'fuse_gap']
    params_changed = False
    old_params = {}
    
    for p in critical_params:
        if f'last_{p}' not in st.session_state: st.session_state[f'last_{p}'] = st.session_state[p]
        old_params[p] = st.session_state[f'last_{p}']
        if st.session_state[f'last_{p}'] != st.session_state[p]:
            params_changed = True
            
    # NOTE: We do NOT update st.session_state[f'last_{p}'] yet! 
    # We wait until after we have attempted to migrate the state.
    
    if st.session_state.elevation_data is None:
        if st.session_state.width_km <= 0 or st.session_state.height_km <= 0:
             st.warning("Please define a valid area (Width and Height must be > 0).")
        else:
            with st.spinner("Fetching elevation data..."):
                try:
                    raw, _ = data_loader.fetch_elevation_data(st.session_state.lat, st.session_state.lon, st.session_state.width_km, st.session_state.height_km, API_KEY)
                    st.session_state.elevation_data = raw
                    st.session_state.island_decisions = {}
                except Exception as e: st.error(str(e))

    if st.session_state.elevation_data is not None:
        settings = project_manager.get_current_settings()
        
        # OPTIMIZATION: Only pass relevant settings to raster processor.
        # This prevents frame/dowel changes from triggering expensive raster re-calculation.
        # CRITICAL: We include CURRENT_CODE_VERSION to force re-calculation when we update the engine 
        # (e.g. adding Super-Resolution or changing blur logic).
        raster_settings = {
            k: settings[k] for k in ['box_d', 'mat_th', 'blur', 'box_w', 'box_h', 'min_area'] if k in settings
        }
        raster_settings['code_version'] = CURRENT_CODE_VERSION
        
        processed_data = geometry_engine.process_terrain_raster(st.session_state.elevation_data, raster_settings)
        
        if processed_data is None:
            st.error("Invalid parameters. Material thickness may be too great for the box depth.")
        else:
            stepped, smooth, n_terrain, m_per_layer, min_elev, max_elev, adjusted_mat_th, max_active_terrain = processed_data
            settings['mat_th'] = adjusted_mat_th
            if not math.isclose(st.session_state.mat_th, adjusted_mat_th):
                st.session_state.show_adjustment_info = True
                st.session_state.adjusted_mat_th = adjusted_mat_th
            
            # --- Robust Layer Navigation Callbacks ---
            def dec_layer():
                st.session_state.layer_idx = max(1, st.session_state.layer_idx - 1)
                _sync_sliders()
            def inc_layer():
                st.session_state.layer_idx = min(n_total, st.session_state.layer_idx + 1)
                _sync_sliders()
            def _sync_sliders():
                idx = st.session_state.layer_idx
                st.session_state.layer_idx_slider_design = idx
                st.session_state.layer_idx_slider_island = idx
                st.session_state.layer_idx_slider_assembly = idx

            # --- INTELLIGENT STATE MANAGEMENT ---
            from utils.state_manager import StateManager
            
            captured_state = None
            if params_changed and not st.session_state.is_new_run:
                with st.spinner("Migrating manual edits..."):
                    try:
                        # 1. Temporarily construct old settings to regenerate the 'previous' state
                        old_settings = settings.copy()
                        for k, v in old_params.items():
                            if k in old_settings: old_settings[k] = v
                        
                        # We also need to recalculate raster properties if box/blur changed
                        # This is expensive but necessary for determining the correct old state
                        old_raster_settings = {k: old_settings[k] for k in ['box_d', 'mat_th', 'blur', 'box_w', 'box_h', 'min_area'] if k in old_settings}
                        old_processed_data = geometry_engine.process_terrain_raster(st.session_state.elevation_data, old_raster_settings)
                        
                        if old_processed_data:
                            _, old_smooth, old_n_terrain, old_m_per_layer, old_min_elev, _, _, _ = old_processed_data
                            # Generate old layers
                            old_raw_geoms = geometry_engine.generate_all_layer_data_v3(old_smooth, old_min_elev, old_m_per_layer, old_n_terrain, old_settings)
                            
                            # Capture!
                            captured_state = StateManager.capture_edit_context(old_raw_geoms)
                        
                    except Exception as e:
                        print(f"State Migration Failed: {e}")
                        # If capture fails, we just proceed with empty state (the old behavior)

            # --- OPTIMIZED PROCESSING ---
            # --- OPTIMIZED PROCESSING ---
            # Version Control for Cache Invalidation
            
            cache_outdated = False
            
            cache_outdated = False
            if st.session_state.get('code_version') != CURRENT_CODE_VERSION:
                cache_outdated = True
                st.session_state.code_version = CURRENT_CODE_VERSION
                # Force clear critical caches
                if 'cached_final_geoms' in st.session_state: del st.session_state.cached_final_geoms
                st.toast("Updated Geometry Engine. Regenerating...", icon="🔄")

            # Only re-compute if critical parameters changed, data is missing, or code was updated
            if 'cached_final_geoms' not in st.session_state or params_changed or processed_data is None or cache_outdated:
                
                with st.spinner("Generating layers..."):
                    # Generate new raw layers
                    # CRITICAL: We inject the code version into the settings dict.
                    # This implies valid hashing for the generation function, forcing it to notice 
                    # that we have 'new code' (or rather, a new request context) and re-run 
                    # instead of returning the cached result for the ignored _smooth_data.
                    settings['code_version'] = CURRENT_CODE_VERSION
                    raw_layer_geoms = geometry_engine.generate_all_layer_data_v3(smooth, min_elev, m_per_layer, n_terrain, settings)
                
                all_layer_geoms = []
                all_modifications = []
                
                progress_bar = st.progress(0, text="Auto-Healing Layers...")
                
                total_layers = len(raw_layer_geoms)
                
                for i in range(total_layers):
                    layer_raw = raw_layer_geoms[i]
                    layer_num = i + 1
                    
                    # Update Progress
                    progress_bar.progress((i + 1) / total_layers, text=f"Auto-Healing Layer {layer_num}/{total_layers}...")
                    
                    # Get support from processing history (previous layer in this loop)
                    layer_below_polys = None
                    if i > 0 and all_layer_geoms:
                        layer_below_polys = [p['poly'] for p in all_layer_geoms[i-1] if not p['poly'].is_empty]

                    healed_layer, mods = geometry_engine.auto_heal_layer(layer_raw, smooth, layer_num, m_per_layer, min_elev, settings, layer_below_polys)
                    
                    all_layer_geoms.append(healed_layer)
                    all_modifications.append(mods)
                
                progress_bar.empty()
                
                # Cache Results
                st.session_state.cached_raw_geoms = raw_layer_geoms
                st.session_state.cached_final_geoms = all_layer_geoms
                st.session_state.cached_modifications = all_modifications
                
                # Update 'last_p' now that we have successfully processed the new state
                for p in critical_params: st.session_state[f'last_{p}'] = st.session_state[p]

            # Use Cached Data
            raw_layer_geoms = st.session_state.cached_raw_geoms # Raw geoms are immutable-ish (we don't modify them in place usually)
            all_layer_geoms = copy.deepcopy(st.session_state.cached_final_geoms) # Deep copy to allow modification by filters
            st.session_state.auto_modifications = st.session_state.cached_modifications # We don't modify these, shallow copy ok? Let's treat as read-only.
            
            
            n_total = len(all_layer_geoms)
            
            # Debug: Print layer sizes to verify persistence
            # for i, l in enumerate(all_layer_geoms):
            #     print(f"Layer {i+1}: {len(l)} polys")
            
            # --- RESTORE STATE ---
            if captured_state:
                restored_count = StateManager.restore_edit_context(captured_state, all_layer_geoms)
                if restored_count > 0:
                    st.toast(f"Restored {restored_count} manual edits!", icon="🧠")
            



            if st.session_state.is_new_run:
                st.session_state.layer_idx = n_total
                st.session_state.is_new_run = False
                st.session_state.layer_idx_slider_design = n_total
                st.session_state.layer_idx_slider_island = n_total
                st.session_state.layer_idx_slider_assembly = n_total

            # Safety clamp: Ensure layer_idx is within valid range (e.g. if n_total changed)
            if st.session_state.layer_idx > n_total: st.session_state.layer_idx = n_total
            
            # Sync slider keys to master layer_idx to ensure UI consistency across tabs
            st.session_state.layer_idx_slider_design = st.session_state.layer_idx
            st.session_state.layer_idx_slider_island = st.session_state.layer_idx
            st.session_state.layer_idx_slider_assembly = st.session_state.layer_idx

            colorscale = get_colorscale('Earth')
            layer_colors = sample_colorscale(colorscale, np.linspace(0, 1, n_total))

            # --- FIX: Recalculate Areas & Re-Filter (Robust) ---
            # 1. Ensure geometry is valid (fix self-intersections).
            # 2. Recalculate area from the polygon (mm) to ensure it matches physical units.
            # 3. Filter based on min_area threshold.
            min_area_threshold = st.session_state.min_area

            for i in range(len(all_layer_geoms)):
                new_layer = []
                for p in all_layer_geoms[i]:
                    if 'poly' in p and not p['poly'].is_empty:
                        # Fix invalid geometries (e.g. bow-ties) that cause area calculation errors
                        if not p['poly'].is_valid:
                            p['poly'] = p['poly'].buffer(0)
                        if p['poly'].is_empty: continue

                        p['area'] = p['poly'].area
                        # Keep if it meets area threshold OR if it's a flagged problem (so we can show it in UI)
                        if p['type'] == 'problem' or p['area'] >= min_area_threshold:
                            new_layer.append(p)
                all_layer_geoms[i] = new_layer

            filtered_geoms = geometry_engine.filter_geometries_based_on_decisions(all_layer_geoms, st.session_state.island_decisions)
            merged_geoms = geometry_engine.apply_merges_to_geometries(filtered_geoms, all_layer_geoms, st.session_state.merge_groups, st.session_state.bridge_thickness, smooth, settings)
            bridged_geoms = geometry_engine.apply_bridges_to_mainland(merged_geoms, all_layer_geoms, st.session_state.bridge_to_mainland_requests, st.session_state.bridge_thickness, smooth, settings)
            final_geoms, bridge_status = geometry_engine.apply_manual_bridges(bridged_geoms, st.session_state.manual_bridge_points, st.session_state.bridge_thickness, settings, smooth)

            # --- VIEW ROUTING ---
            if st.session_state.current_view == "3D Design":
                views.render_design_view(final_geoms, settings, st.session_state.layer_idx, n_total, all_layer_geoms)

            elif st.session_state.current_view == "Layer Tools":
                views.render_layer_tools(st.session_state.layer_idx, n_total, all_layer_geoms, final_geoms, layer_colors, bridge_status, raw_layer_geoms)

            elif st.session_state.current_view == "Assembly Tools":
                current_dowels = []
                # Dowel logic archived
                views.render_assembly_tools(st.session_state.layer_idx, n_total, final_geoms, [])

            elif st.session_state.current_view == "Preview":
                views.render_preview(final_geoms, settings, n_total, all_layer_geoms)
            
            elif st.session_state.current_view == "Export":
                current_dowels = []
                # Dowel logic archived
                views.render_export(final_geoms, settings, [])