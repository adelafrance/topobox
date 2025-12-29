# project_manager.py

import json
import os
import streamlit as st
from utils import github_storage

PROJECTS_DIR = "projects"
SUBMISSIONS_DIR = "submissions"

for d in [PROJECTS_DIR, SUBMISSIONS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def list_projects():
    """Returns a sorted list of available project names."""
    return sorted([f.replace('.json', '') for f in os.listdir(PROJECTS_DIR) if f.endswith('.json')])

def list_submissions(load_from_drive=True):
    """Returns a list of submissions from the Local Folder (Synced via Git)."""
    # Note: 'load_from_drive' arg is kept for compatibility but ignored.
    # The 'Cloud' is now the Git Repo, which the user must 'Pull' to see here.
    local_files = set()
    if os.path.exists(SUBMISSIONS_DIR):
        local_files = set(f.replace('.json', '') for f in os.listdir(SUBMISSIONS_DIR) if f.endswith('.json'))
    
    return sorted(list(local_files))

def submit_design(name, state, submission_info=None):
    """Saves to Local and Pushes to GitHub Repo."""
    safe_name, save_data = _prepare_save_data(name, state)
    
    # Inject Name explicitly so we can list it later
    save_data['project_name'] = safe_name 
    
    # Inject Submission Info
    if submission_info:
        save_data['submission_info'] = submission_info
        
    filename = f"{safe_name}.json" # Cleaned up naming (no _SUBMISSION suffix needed if in folders, but let's keep it compatible?)
    # actually, previous logic enforced _SUBMISSION.json. Let's keep that for consistency if we want.
    # But wait, GitHub storage is generic. 
    # Let's stick to the convention: {name}_SUBMISSION.json
    filename = f"{safe_name}_SUBMISSION.json"
    filepath = os.path.join(SUBMISSIONS_DIR, filename)
    
    # Local Save (Backup / Maker Logic)
    try:
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=4)
    except Exception as e:
        # If running on Cloud (read-only), this might fail or be transient. That's OK.
        print(f"Local save warning: {e}")
        
    # Cloud Push (GitHub)
    upload_error = None
    if st.secrets.get("github"):
        err = github_storage.push_json(filename, save_data, message=f"Submission: {safe_name}")
        if err:
            upload_error = err
            print(f"GitHub Push Error: {err}")
    else:
        # If no secrets (e.g. Maker without internet?), just local is fine.
        pass
        
    return filename, upload_error

def load_submission(name):
    """Loads from Local Submissions Folder."""
    filename = f"{name}_SUBMISSION.json"
    filepath = os.path.join(SUBMISSIONS_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
            
    return None



def get_project_json(name, state):
    """Returns the JSON string for the project."""
    safe_name, save_data = _prepare_save_data(name, state)
    return json.dumps(save_data, indent=4), safe_name

def save_project(name, state):
    """Saves the current settings to a nested JSON file."""
    safe_name, save_data = _prepare_save_data(name, state)
    filepath = os.path.join(PROJECTS_DIR, f"{safe_name}.json")
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=4)
    return filepath

def delete_submission(name_or_file):
    """Deletes a submission from GitHub Repo AND Local Disk. Auto-resolves extension."""
    errors = []
    
    # Resolve Filename
    # Try exact, then .json, then _SUBMISSION.json
    candidates = [name_or_file, f"{name_or_file}.json", f"{name_or_file}_SUBMISSION.json"]
    local_target = None
    
    # Find Local Target
    for c in candidates:
        path = os.path.join(SUBMISSIONS_DIR, c)
        if os.path.exists(path):
            local_target = c
            break
            
    # If not found locally, default to appending .json for remote attempt? 
    # Or just use the input if it looks like a file?
    target_filename = local_target if local_target else candidates[2] # Guess _SUBMISSION.json preferrably
    
    # 1. Remote Delete
    if st.secrets.get("github"):
        # We might need to try multiple if we aren't sure, but usually local matches remote.
        err = github_storage.delete_file(target_filename, message=f"Deleted {target_filename}")
        if err:
            errors.append(f"Remote: {err}")
            
    # 2. Local Delete (ALWAYS RUN)
    if local_target:
        filepath = os.path.join(SUBMISSIONS_DIR, local_target)
        try:
             os.remove(filepath)
        except Exception as e:
            errors.append(f"Local: {e}")
    else:
        # If not found locally, we can't delete it.
        # But if it was a ghost entry, maybe we should just say success?
        pass # It's gone
        
    return "; ".join(errors) if errors else None

def _prepare_save_data(name, state):
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_name:
        safe_name = "Untitled Project"
    
    save_data = {
        "location": {
            "lat": state.get("lat"),
            "lon": state.get("lon"),
            "width_km": state.get("width_km"),
            "height_km": state.get("height_km")
        },
        "box": {
            "width": state.get("box_w"),
            "height": state.get("box_h"),
            "depth": state.get("box_d")
        },
        "material": {
            "thickness": state.get("mat_th")
        },
        "processing": {
            "blur": state.get("blur"),
            "min_area": state.get("min_area"),
            "min_feature_width": state.get("min_feature_width"),
            "auto_bridge": state.get("auto_bridge"),
            "auto_cleanup": state.get("auto_cleanup"),
            "auto_fuse": state.get("auto_fuse"),
            "fuse_gap": state.get("fuse_gap")
        },
        "frame": {
            "mode": state.get("frame_mode"),
            "width": state.get("frame_width"),
            "sides": state.get("frame_sides")
        },
        "jig": {
            "conn_width": state.get("jig_conn_width"),
            "grid_spacing": state.get("jig_grid_spacing"),
            "fluid": state.get("jig_fluid")
        },
        "dowels": {
            "use": state.get("use_dowels"),
            "diameter": state.get("dowel_diam"),
            "count": state.get("num_dowels"),
            "positions": []
        },
        "manual_edits": {
            "merge_groups": state.get("merge_groups", {}),
            "merge_group_names": state.get("merge_group_names", {}),
            "manual_bridge_points": state.get("manual_bridge_points", {}),
            "island_decisions": state.get("island_decisions", {}),
            "jig_modifications": state.get("jig_modifications", {}),
            "problem_decisions": state.get("problem_decisions", {}),
            "manual_problems": state.get("manual_problems", {})
        },
        "project_name": state.get("proj_name")
    }

    if state.get('use_dowels') and state.get('num_dowels'):
        for i in range(state.get('num_dowels')):
            save_data['dowels']['positions'].append({
                'x': state.get(f'dowel_{i}_x'),
                'y': state.get(f'dowel_{i}_y'),
                'skip': state.get(f'dowel_{i}_skip')
            })

    return safe_name, save_data

def load_project_from_json(json_data):
    """Loads settings from a dict."""
    return json_data

def load_project(name):
    """Loads a project JSON file from disk."""
    filepath = os.path.join(PROJECTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error loading project file: {e}")
        return None

def apply_settings(data):
    """
    Applies settings from a loaded (and potentially nested) project file
    to the flat structure of Streamlit's session_state.
    """
    # Location
    loc = data.get("location", {})
    st.session_state.lat = loc.get("lat", 45.976)
    st.session_state.lon = loc.get("lon", 7.658)
    st.session_state.width_km = loc.get("width_km", 50.0)
    st.session_state.height_km = loc.get("height_km", 50.0)

    # Box
    box = data.get("box", {})
    st.session_state.box_w = box.get("width", 200.0)
    st.session_state.box_h = box.get("height", 200.0)
    st.session_state.box_d = box.get("depth", 40.0)

    # Material
    mat = data.get("material", {})
    st.session_state.mat_th = mat.get("thickness", 2.0)

    # Processing
    proc = data.get("processing", {})
    st.session_state.blur = proc.get("blur", 1.0)
    st.session_state.min_area = proc.get("min_area", 10.0)
    st.session_state.min_feature_width = proc.get("min_feature_width", 3.0)
    st.session_state.auto_bridge = proc.get("auto_bridge", True)
    st.session_state.auto_cleanup = proc.get("auto_cleanup", True)
    st.session_state.auto_fuse = proc.get("auto_fuse", True)
    st.session_state.fuse_gap = proc.get("fuse_gap", 1.0)
    
    # Frame
    frame = data.get("frame", {})
    st.session_state.frame_mode = frame.get("mode", "None")
    st.session_state.frame_width = frame.get("width", 4.0)
    st.session_state.frame_sides = frame.get("sides", {'top': True, 'bottom': True, 'left': True, 'right': True})

    # Jig
    jig = data.get("jig", {})
    st.session_state.jig_conn_width = jig.get("conn_width", 6.0)
    st.session_state.jig_grid_spacing = jig.get("grid_spacing", 20.0)
    st.session_state.jig_fluid = jig.get("fluid", True)

    # Dowels
    dowels = data.get("dowels", {})
    st.session_state.use_dowels = dowels.get("use", True)
    st.session_state.dowel_diam = dowels.get("diameter", 3.0)
    st.session_state.num_dowels = dowels.get("count", 2)
    positions = dowels.get("positions", [])
    for i, pos in enumerate(positions):
        if i < 4: # Hard limit of 4 dowels
            st.session_state[f'dowel_{i}_x'] = pos.get('x')
            st.session_state[f'dowel_{i}_y'] = pos.get('y')
            st.session_state[f'dowel_{i}_skip'] = pos.get('skip')

    # Project Name
    st.session_state.proj_name = data.get("project_name", "Loaded Project")

    # Manual Edits
    edits = data.get("manual_edits", {})
    st.session_state.merge_groups = edits.get("merge_groups", {})
    st.session_state.merge_group_names = edits.get("merge_group_names", {})
    st.session_state.manual_bridge_points = edits.get("manual_bridge_points", {})
    st.session_state.island_decisions = edits.get("island_decisions", {})
    st.session_state.jig_modifications = edits.get("jig_modifications", {})
    st.session_state.problem_decisions = edits.get("problem_decisions", {})
    st.session_state.manual_problems = edits.get("manual_problems", {})

    # Reset non-persistent runtime state
    st.session_state['elevation_data'] = None
    
    # --- SYNC UI WIDGETS ---
    # We must explicitly set the 'inputs' that feed the state, otherwise they will show defaults/stale values
    # Lat/Lon Input String
    st.session_state.coords_input = f"{st.session_state.lat:.4f}, {st.session_state.lon:.4f}"
    
    # Blur Slider/Input Sync
    st.session_state.blur_slider = st.session_state.blur
    st.session_state.blur_input = st.session_state.blur
    
    # --- TRIGGER AUTO-RUN ---
    # Force the app to treat this as a "Go" signal
    st.session_state.run_btn = True
    st.session_state.is_new_run = True # Resets layer indexing
    
    # Versioning
    st.session_state.code_version = "LOADED_STATE" # Force refresh check if needed
    st.session_state['is_new_run'] = True
    st.session_state['run_btn'] = True # Trigger a re-run

def get_current_settings():
    """
    Gathers settings from the flat session_state into a clean, nested dictionary
    that will be used by the geometry and visualization engines.
    This is the inverse of apply_settings.
    """
    settings = {
        "lat": st.session_state.get("lat"),
        "lon": st.session_state.get("lon"),
        "width_km": st.session_state.get("width_km"),
        "height_km": st.session_state.get("height_km"),
        "box_w": st.session_state.get("box_w"),
        "box_h": st.session_state.get("box_h"),
        "box_d": st.session_state.get("box_d"),
        "mat_th": st.session_state.get("mat_th"),
        "blur": st.session_state.get("blur"),
        "min_area": st.session_state.get("min_area"),
        "min_area": st.session_state.get("min_area"),
        "min_feature_width": st.session_state.get("min_feature_width", 3.0),
        "auto_bridge": st.session_state.get("auto_bridge", True),
        "auto_cleanup": st.session_state.get("auto_cleanup", True),
        "auto_fuse": st.session_state.get("auto_fuse", True),
        "fuse_gap": st.session_state.get("fuse_gap", 0.5),
        "frame_mode": st.session_state.get("frame_mode"),
        "frame_width": st.session_state.get("frame_width"),
        "frame_sides": st.session_state.get("frame_sides"),
        "use_dowels": st.session_state.get("use_dowels"),
        "dowel_diam": st.session_state.get("dowel_diam"),
        "num_dowels": st.session_state.get("num_dowels"),
        "proj_name": st.session_state.get("proj_name"),
        "jig_conn_width": st.session_state.get("jig_conn_width", 6.0),
        "jig_grid_spacing": st.session_state.get("jig_grid_spacing", 20.0),
        "jig_fluid": st.session_state.get("jig_fluid", True),
        "dowel_data": []
    }
    
    # Dowels Archived - Force Empty
    settings['use_dowels'] = False
    
    # if settings.get('use_dowels') and settings.get('num_dowels'):
    #     for i in range(settings['num_dowels']):
    #         settings['dowel_data'].append({
    #             'x': st.session_state.get(f'dowel_{i}_x'),
    #             'y': st.session_state.get(f'dowel_{i}_y'),
    #             'skip': st.session_state.get(f'dowel_{i}_skip')
    #         })
    return settings