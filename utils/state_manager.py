
import streamlit as st
import numpy as np
import math
from shapely.geometry import Point, box, Polygon
from shapely.ops import nearest_points

class StateManager:
    """
    Intelligent State Persistence for TopoBox.
    
    This class handles the "Memory" of the application. It captures the 'essence' of manual
    modifications (where they are geometrically) rather than just their index (which changes).
    When the terrain is regenerated (due to blur/resize changes), this class attempts to
    re-apply those modifications to the new geometry.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def capture_edit_context(all_layer_geoms):
        """
        Captures the geometric context of all current manual edits.
        Returns a dictionary containing the 'essence' of the state.
        """
        if not all_layer_geoms: return {}

        context = {
            'merge_groups': {},      # list of {center: (x,y), layer: L, original_indices: []}
            'bridge_requests': {},   # list of {center: (x,y), layer: L}
            'problem_decisions': {}, # list of {centroid: (x,y), layer: L, action: ...}
            'manual_problems': {}    # list of {box: (x,y,w,h), layer: L}
        }

        # 1. Capture Merge Groups
        # We store the centroid of the simplified union of the group.
        # This helps us find the same cluster of islands later.
        for layer_key, groups in st.session_state.merge_groups.items():
            layer_idx = int(layer_key.lstrip('L')) - 1
            if layer_idx < 0 or layer_idx >= len(all_layer_geoms): continue
            
            layer_islands = [p for p in all_layer_geoms[layer_idx] if p['type'] == 'island']
            
            context['merge_groups'][layer_key] = []
            for group_indices in groups:
                # Find geometric center of this group
                group_polys = [layer_islands[i]['poly'] for i in group_indices if i < len(layer_islands)]
                if not group_polys: continue
                
                # We use the centroid of the first item as the anchor, or average of all
                # Simple average of centroids is robust enough
                cx = np.mean([p.centroid.x for p in group_polys])
                cy = np.mean([p.centroid.y for p in group_polys])
                context['merge_groups'][layer_key].append({'center': (cx, cy)})

        # 2. Capture Bridge to Mainland Requests
        for layer_key, indices in st.session_state.bridge_to_mainland_requests.items():
            layer_idx = int(layer_key.lstrip('L')) - 1
            if layer_idx < 0 or layer_idx >= len(all_layer_geoms): continue
            
            layer_islands = [p for p in all_layer_geoms[layer_idx] if p['type'] == 'island']
            
            context['bridge_requests'][layer_key] = []
            for idx in indices:
                if idx < len(layer_islands):
                    p = layer_islands[idx]['poly']
                    context['bridge_requests'][layer_key].append({'center': (p.centroid.x, p.centroid.y)})

        # 3. Capture Problem Decisions (The hardest part)
        # Problems are indexed by P{idx}. We need to store their location to re-find them.
        for key, decision in st.session_state.problem_decisions.items():
            # key is L{layer}_P{idx} or L{layer}_M{idx}
            parts = key.split('_')
            if len(parts) < 2: continue
            layer_str, id_str = parts[0], parts[1]
            layer_idx = int(layer_str.lstrip('L')) - 1
            
            if layer_idx < 0 or layer_idx >= len(all_layer_geoms): continue
            
            if id_str.startswith('P'):
                # It's an auto-detected problem. Find its geometry in the OLD list.
                try:
                    p_idx = int(id_str.lstrip('P'))
                    problems = [p for p in all_layer_geoms[layer_idx] if p['type'] == 'problem']
                    if p_idx < len(problems):
                        poly = problems[p_idx]['poly']
                        context['problem_decisions'][key] = {
                            'center': (poly.centroid.x, poly.centroid.y),
                            'area': poly.area,
                            'decision': decision,
                            'layer_key': layer_str
                        }
                except: pass
            
            # Manual problems (M) are easier, they are defined by a box in 'manual_problems' state
            # We don't need to capture them here specifically if we restore the box list below.
        
        # 4. Capture Manual Problems (The input boxes)
        # These are just coordinates, so they are robust! We just copy them.
        context['manual_problems'] = st.session_state.manual_problems.copy()
        
        # 5. Capture Manual Bridge Points
        # Also robust coordinates.
        context['manual_bridge_points'] = st.session_state.manual_bridge_points.copy()
        
        # 6. Capture Jig Modifications
        context['jig_modifications'] = st.session_state.jig_modifications.copy()

        return context

    @staticmethod
    def restore_edit_context(context, new_layer_geoms):
        """
        Restores the state by mapping the captured 'essence' onto the new geometry.
        """
        if not context or not new_layer_geoms: return
        
        # Reset current state containers
        st.session_state.merge_groups = {}
        st.session_state.bridge_to_mainland_requests = {}
        st.session_state.problem_decisions = {}
        
        # Robust copies (Coordinates don't need re-mapping)
        st.session_state.manual_problems = context.get('manual_problems', {})
        st.session_state.manual_bridge_points = context.get('manual_bridge_points', {})
        st.session_state.jig_modifications = context.get('jig_modifications', {})
        
        restored_count = 0
        
        # 1. Restore Merge Groups
        for layer_key, groups_data in context.get('merge_groups', {}).items():
            layer_idx = int(layer_key.lstrip('L')) - 1
            if layer_idx >= len(new_layer_geoms): continue
            
            current_islands = [p for p in new_layer_geoms[layer_idx] if p['type'] == 'island']
            if not current_islands: continue
            
            st.session_state.merge_groups[layer_key] = []
            
            for group_data in groups_data:
                cx, cy = group_data['center']
                target = Point(cx, cy)
                
                # Find all islands "close enough" to this center to be part of the group
                # Strategy: We look for the nearest islands.
                # Since a merge group connects islands, we expect islands to be near this centroid.
                
                candidates = []
                for i, island in enumerate(current_islands):
                    # Distance to bounding box or centroid?
                    dist = island['poly'].distance(target)
                    if dist < 50.0: # Search radius (mm) - Generous but safe
                         candidates.append((dist, i))
                
                candidates.sort(key=lambda x: x[0])
                # We need at least 2 to make a group. 
                # Let's take the ones that are very likely the same ones.
                if len(candidates) >= 2:
                    # In a perfect restore, the indices match. In a rough restore, we guess.
                    # We pick top k? No, we don't know k.
                    # We pick islands that are within a tight radius of the original center?
                    # Better heuristic: Check if the island actually CONTAINED the center? 
                    # Or check intersection?
                    
                    # Refined Heuristic: Take all candidates within standard connection distance (e.g. 10mm)
                    # of each other? No.
                    
                    # Let's just take the closest 2 or 3, assuming the user merged neighbors.
                    # We accept the top candidates that are 'close' (e.g. < 20mm from center)
                    final_indices = [idx for d, idx in candidates if d < 20.0]
                    if len(final_indices) >= 2:
                        st.session_state.merge_groups[layer_key].append(sorted(final_indices))
                        restored_count += 1

        # 2. Restore Bridge Requests
        for layer_key, requests in context.get('bridge_requests', {}).items():
            layer_idx = int(layer_key.lstrip('L')) - 1
            if layer_idx >= len(new_layer_geoms): continue
            
            current_islands = [p for p in new_layer_geoms[layer_idx] if p['type'] == 'island']
            
            st.session_state.bridge_to_mainland_requests[layer_key] = []
            for req in requests:
                cx, cy = req['center']
                target = Point(cx, cy)
                
                # Find the single closest island
                best_idx = -1
                best_dist = float('inf')
                
                for i, island in enumerate(current_islands):
                    dist = island['poly'].distance(target)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                
                # If it's reasonably close (it should be nearly 0 if it's the same island), match it
                if best_idx != -1 and best_dist < 10.0:
                    if best_idx not in st.session_state.bridge_to_mainland_requests[layer_key]:
                        st.session_state.bridge_to_mainland_requests[layer_key].append(best_idx)
                        restored_count += 1

        # 3. Restore Problem Decisions
        for key, data in context.get('problem_decisions', {}).items():
            # If it was manual (M), it persists via manual_problems list. But we need to keep the decision (action).
            # The manual_problems list only stores the box. The decision is in problem_decisions.
            
            if '_M' in key:
                # Manual problem keys are L{N}_M{idx}. 
                # The index might have shifted if we deleted some? 
                # Actually manual_problems is a list, so indices correspond.
                # If we copied manual_problems exactly, the indices `M0`, `M1` are valid.
                st.session_state.problem_decisions[key] = data['decision']
                continue
            
            # Auto-detected problems (P)
            # We need to find the new problem that matches the old location.
            layer_str = data['layer_key']
            layer_idx = int(layer_str.lstrip('L')) - 1
            if layer_idx >= len(new_layer_geoms): continue
            
            new_problems = [p for p in new_layer_geoms[layer_idx] if p['type'] == 'problem']
            
            target = Point(data['center'])
            
            best_p_idx = -1
            best_dist = float('inf')
            
            for i, prob in enumerate(new_problems):
                # Distance between centroids
                dist = prob['poly'].centroid.distance(target)
                if dist < best_dist:
                    best_dist = dist
                    best_p_idx = i
            
            # If we found a match close enough (e.g. < 5mm shift), apply the decision
            if best_p_idx != -1 and best_dist < 10.0:
                new_key = f"{layer_str}_P{best_p_idx}"
                st.session_state.problem_decisions[new_key] = data['decision']
                restored_count += 1

        return restored_count
