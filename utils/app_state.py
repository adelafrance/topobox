import streamlit as st
import os
import pickle

AUTOSAVE_FILE = "autosave.pkl"

def initialize_session_state():
    """Initializes all default session state variables."""
    defaults = {
        'lat': 45.976, 'lon': 7.658, 'width_km': 15.0, 'height_km': 15.0,
        'box_w': 190.0, 'box_h': 190.0, 'box_d': 30.0, 'mat_th': 2.0, 'blur': 1.0,
        'min_area': 10.0, 'min_feature_width': 3.0, 'frame_mode': "None", 'frame_width': 4.0,
        'frame_sides': {'top': True, 'bottom': True, 'left': True, 'right': True},
        # --- ARCHIVED FEATURE: DOWELS ---
        # Dowels are explicitly SUPPRESSED. Do not enable. 
        # Future confusion avoidance: "There should absolutely be no dowels."
        'use_dowels': False, 'dowel_diam': 3.0, 'num_dowels': 2,
        'layer_idx': 1, 'is_new_run': True, 'camera_snap_design': None, 'camera_snap_preview': None,
        'island_decisions': {}, 'problem_decisions': {}, 'manual_problems': {}, 'merge_group_names': {}, 
        'elevation_data': None, 'proj_name': "MyMountain_v1",
        'show_slicer': False, 'run_btn': False,
        'quit_confirmation_visible': False, 'show_adjustment_info': False,
        'adjusted_mat_th': 0.0, 'show_layer_below': False, 'merge_groups': {},
        'bridge_thickness': 2.0, 'bridge_to_mainland_requests': {}, 'manual_bridge_points': {},
        'show_3d_axes': False, 'item_to_delete': None, 'editing_manual_bridge': None,
        'show_original_islands': False, 'jig_modifications': {},
        'jig_anchor_thick': 6.0, 'jig_anchor_len': 25.0,
        'current_view': "Home",
        'auto_cleanup': True, 'auto_bridge': True, 'auto_fuse': True, 'fuse_gap': 1.0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Initialize dowels
    for i in range(4):
        st.session_state.setdefault(f'dowel_{i}_x', 50.0 + (i*30))
        st.session_state.setdefault(f'dowel_{i}_y', 50.0 + (i*30))
        st.session_state.setdefault(f'dowel_{i}_skip', 1)

def restore_autosave():
    """Restores state from pickle file if available and not already initialized."""
    if 'lat' not in st.session_state and os.path.exists(AUTOSAVE_FILE):
        try:
            with open(AUTOSAVE_FILE, 'rb') as f:
                saved_state = pickle.load(f)
                for k, v in saved_state.items():
                    # Filter out button keys AND current_view (Always start Home)
                    if k != 'current_view' and not k.endswith(('_btn', '_btn_design', '_btn_island', '_btn_prev', '_btn_assembly')) and not k.startswith(('del_', 'island_view_', 'island_plot_', 'regen_', 'assembly_plot_')):
                        st.session_state[k] = v
            
            st.session_state.quit_confirmation_visible = False
            st.session_state.is_new_run = True
            st.toast("Session restored!", icon="🔄")
        except Exception: pass

def save_autosave():
    """Saves current state to pickle file."""
    try:
        with open(AUTOSAVE_FILE, 'wb') as f:
            state_to_save = {k: v for k, v in st.session_state.items() 
                             if k != 'current_view' and not k.endswith(('_btn', '_btn_design', '_btn_island', '_btn_prev', '_btn_assembly')) and not k.startswith(('del_', 'island_view_', 'island_plot_', 'regen_', 'assembly_plot_'))}
            pickle.dump(state_to_save, f)
    except Exception: pass

def run_migrations():
    """Ensures state values are valid and migrated from older versions."""
    if st.session_state.get('jig_offset') == 8.0: st.session_state['jig_offset'] = 6.0
    if st.session_state.get('jig_conn_width') == 16.0: st.session_state['jig_conn_width'] = 6.0
    if st.session_state.get('jig_offset') == 1.0: st.session_state['jig_offset'] = 6.0
    if st.session_state.get('jig_conn_width') == 1.0: st.session_state['jig_conn_width'] = 6.0
    if st.session_state.get('jig_anchor_thick') == 1.0: st.session_state['jig_anchor_thick'] = 6.0
    if st.session_state.get('jig_anchor_len') in [10.0, 40.0]: st.session_state['jig_anchor_len'] = 25.0

def handle_deferred_deletion():
    """Processes any deletion requests queued in session state."""
    if st.session_state.get('item_to_delete'):
        info = st.session_state.item_to_delete
        l_key, index, type_ = info['layer'], info['index'], info['type']
        
        target_map = {
            'merge': st.session_state.merge_groups,
            'bridge': st.session_state.bridge_to_mainland_requests,
            'manual': st.session_state.manual_bridge_points,
            'jig_mod': st.session_state.jig_modifications,
            'manual_prob': st.session_state.manual_problems
        }
        
        if type_ in target_map and l_key in target_map[type_]:
            collection = target_map[type_][l_key]
            if index < len(collection):
                collection.pop(index)
                if not collection:
                    del target_map[type_][l_key]
        
        st.session_state.item_to_delete = None