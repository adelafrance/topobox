# geometry_engine.py

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage import measure, morphology
from skimage.graph import route_through_array
from shapely.geometry import Polygon, MultiPolygon, box, Point, LineString
from shapely.ops import unary_union, nearest_points
from shapely.validation import make_valid
from shapely import affinity
from scipy.spatial import Delaunay
import streamlit as st
import uuid
import svgwrite
from io import StringIO
import math
import copy
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

def clean_geometry(geom, min_width=3.0):
    """
    Removes features thinner than min_width using morphological opening.
    This prevents thin, fragile parts that might burn during laser cutting.
    Returns the cleaned geometry and the parts that were removed.
    """
    if geom is None or geom.is_empty: return geom, None
    
    # Optimization: Simplify before expensive buffering
    # 0.05mm tolerance is well below laser kerf (0.2mm) so visual impact is zero.
    geom = geom.simplify(0.05, preserve_topology=True)
    
    # Buffer negative then positive (Opening)
    # We use slightly less than half width (0.495) to preserve features that are exactly the min width.
    cleaned = geom.buffer(-min_width * 0.495).buffer(min_width * 0.495)
    removed = geom.difference(cleaned)
    return cleaned, removed

def analyze_thin_features(geom, min_width=3.0, auto_cleanup=True):
    """
    Smartly identifies thin features.
    - 'Protrusions' (connected at 1 spot) are kept (merged back).
    - 'Bridges' (connected at >=2 spots) are flagged as problems.
    """
    if geom is None or geom.is_empty: return geom, []
    
    # 1. Create robust base (remove all thin stuff)
    # Use high resolution buffers to prevent angular artifacts
    eroded = geom.buffer(-min_width * 0.495, join_style=1, resolution=32)
    cleaned = eroded.buffer(min_width * 0.495, join_style=1, resolution=32)
    
    # 2. Find what was removed
    diff = geom.difference(cleaned)
    if diff.is_empty: 
        # If nothing was removed, there are no thin features to analyze.
        # Return original geom and empty problem list.
        return geom, []
    
    thin_polys = list(diff.geoms) if diff.geom_type == 'MultiPolygon' else [diff]
    
    bridges = []
    safe_protrusions = []
    
    for p in thin_polys:
        # Check connections to the robust base
        # Buffer slightly (0.1mm) to detect touch points
        intersection = p.buffer(0.1).intersection(cleaned)
        
        # Count distinct connection areas
        parts = list(intersection.geoms) if intersection.geom_type in ['MultiPolygon', 'GeometryCollection'] else [intersection]
        # Filter out tiny noise intersections
        parts = [x for x in parts if x.area > 0.01]
        
        if len(parts) >= 2: 
            bridges.append(p)      # Connects 2+ masses -> Bridge (Flag it)
        else:
            # Connects 1 mass -> Protrusion
            # If it is significant in size (e.g. a "lollipop" or long ridge), flag it.
            # Threshold: Lowered to 0.25 * width^2 to catch smaller "lollipops"
            if p.area > (min_width * min_width * 0.25):
                bridges.append(p)
            else: 
                safe_protrusions.append(p)       # Tiny spike -> Auto-keep
    
    # Re-assemble: Base + Safe Protrusions
    final_geom = unary_union([cleaned] + safe_protrusions)
    return final_geom, bridges

@st.cache_data
def process_terrain_raster(elevation_data, settings):
    # --- SUPER-RESOLUTION UPSCALING ---
    # Low-res input creates "pixel step" jaggedness in contours.
    # We enforce a high target resolution (e.g. ~2000px) to ensure smooth curves.
    target_dim = 1200.0  # Optimized for Cloud Performance (was 2000.0)
    h, w = elevation_data.shape
    scale_factor = 1.0
    
    # --- 1. PRE-SMOOTHING (Blur Low-Res) ---
    # We blur the raw data *before* upscaling.
    # This prevents the cubic interpolator from "ringing" (overshooting) on sharp pixel steps.
    # We enforce a small minimum blur (0.8px) if we plan to upscale, because cubic on sharp steps is bad.
    base_blur = max(settings['blur'], 0.8) if (max(h, w) < target_dim) else settings['blur']
    data_smooth = gaussian_filter(elevation_data, sigma=base_blur)

    # --- 2. SUPER-RESOLUTION UPSCALING ---
    if max(h, w) < target_dim:
        scale_factor = target_dim / max(h, w)
        # Now we safely upscale the smooth blob using Cubic interpolation.
        # This gives us perfect high-res curves without artifacts.
        data_smooth = zoom(data_smooth, scale_factor, order=3, mode='nearest')
    
    box_d, mat_th = settings['box_d'], settings['mat_th']
    if mat_th <= 0 or box_d < mat_th: return None
    theoretical_layers = box_d / mat_th
    num_total_layers = int(round(theoretical_layers)) if math.isclose(theoretical_layers, round(theoretical_layers)) else int(math.floor(theoretical_layers))
    if num_total_layers < 2: return None
    adjusted_mat_th = box_d / num_total_layers
    num_terrain_layers = num_total_layers
    
    num_terrain_layers = num_total_layers
    
    # (Blur is already applied)
    # data_smooth variable exists.
    
    min_elev, max_elev = np.min(data_smooth), np.max(data_smooth)
    
    px_area = (settings['box_w'] / data_smooth.shape[1]) * (settings['box_h'] / data_smooth.shape[0])
    min_pixels = int(settings['min_area'] / px_area) if px_area > 0 else 0

    # Determine Effective Max Elevation:
    # Scan downwards from the absolute peak to find the first level where a terrain profile
    # (connected component) is larger than min_area.
    effective_max_elev = min_elev
    if min_pixels > 0:
        test_levels = np.linspace(max_elev, min_elev, 100)
        for level in test_levels:
            mask = data_smooth >= level
            labeled = measure.label(mask)
            if any(r.area >= min_pixels for r in measure.regionprops(labeled)):
                effective_max_elev = level
                break
    else:
        effective_max_elev = max_elev

    # Recalculate span based on the effective top, cropping the tiny peak.
    span = effective_max_elev - min_elev
    meters_per_layer = span / num_terrain_layers if num_terrain_layers > 0 else span

    stepped_data = np.full_like(data_smooth, min_elev)
    max_active_i = 0
    for i in range(1, num_terrain_layers + 1):
        level_elev = min_elev + (i * meters_per_layer)
        mask = data_smooth >= level_elev
        if min_pixels > 1: mask = morphology.remove_small_objects(mask, min_size=min_pixels)
        if np.any(mask): max_active_i = i
        stepped_data[mask] = level_elev
    return stepped_data, data_smooth, num_terrain_layers, meters_per_layer, min_elev, max_elev, adjusted_mat_th, max_active_i

@st.cache_data
def generate_all_layer_data_v3(_smooth_data, _min_elev, _m_per_layer, _n_terrain, settings):
    all_layers = []
    n_total = _n_terrain
    
    for i in range(1, n_total + 1):
        is_top = (i == n_total)
        terrain_polys = _generate_terrain_geometry(i, _smooth_data, _min_elev, _m_per_layer, settings, is_top_layer=is_top)
        frame_polys = _generate_frame_geometry(settings)
        combined_polys = [p['poly'] for p in terrain_polys] + [p['poly'] for p in frame_polys]
        
        if not combined_polys:
            all_layers.append([])
        else:
            merged_geom = unary_union(combined_polys)
            final_geom = _apply_dowels_to_geom(merged_geom, settings, i, n_total)
            
            # --- Auto-Fusion (Gap Closing) ---
            # Merges "essentially touching" parts to prevent trivial islands (e.g. < 0.4mm gap)
            if not settings.get('fast_preview', False) and settings.get('auto_fuse', True):
                 final_geom = fuse_close_polygons(final_geom, gap=settings.get('fuse_gap', 0.4))
            
            # 3. Structural Optimization (Bridge fusing)
            # We detect thin "isthmuses" that might break and fuse them.
            # Also detects floating islands and bridges them to mainland.
            if not settings.get('fast_preview', False):
                # Pre-simplify to speed up topology checks
                if settings.get('pre_simplify', True):
                    final_geom = final_geom.simplify(0.05, preserve_topology=True)

                cleaned_geom, problem_list = analyze_thin_features(final_geom, min_width=settings.get('min_feature_width', 3.0), auto_cleanup=settings.get('auto_cleanup', True))
            else:
                # Skip heavy structural/healing logic in preview
                cleaned_geom = final_geom.simplify(0.05, preserve_topology=True)
                problem_list = []
            
            final_polys_list = _classify_polygons(cleaned_geom, settings)
            
            # --- Auto-Bridge (Intelligent Automation) ---
            if not settings.get('fast_preview', False) and settings.get('auto_bridge', False):
                 final_polys_list = auto_bridge_islands(final_polys_list, settings.get('bridge_thickness', 2.0), _smooth_data, settings)
            
            if problem_list:
                # Assign Robust Centroid-Based ID
                # This ensures that even if the list order changes (due to filtering instability), 
                # the ID for a specific geometry location remains constant.
                # Format: "X{int}_Y{int}" (mm resolution effectively)
                
                final_problems = []
                for g in problem_list:
                    if g.is_empty: continue
                    cx, cy = int(g.centroid.x * 10), int(g.centroid.y * 10) # 0.1mm precision hash
                    pid = f"X{cx}_Y{cy}"
                    final_problems.append({'poly': g, 'type': 'problem', 'area': g.area, 'orig_idx': pid})
                
                final_polys_list.extend(final_problems)
                
            all_layers.append(final_polys_list)

    return all_layers

def _generate_terrain_geometry(layer_index, data_smooth, min_elev, meters_per_layer, settings, is_top_layer=False):
    width_mm, height_mm = settings['box_w'], settings['box_h']
    frame_poly_shape = box(0, 0, width_mm, height_mm)
    if layer_index == 1: return [{'poly': frame_poly_shape, 'type': 'base', 'area': width_mm * height_mm}]
    terrain_idx = layer_index - 1
    level_value = min_elev + ((terrain_idx + 0.5) * meters_per_layer)
    padded_data = np.pad(data_smooth, pad_width=1, mode='constant', constant_values=np.min(data_smooth)-100)
    contours = measure.find_contours(padded_data, level_value)
    scale_x, scale_y = width_mm / data_smooth.shape[1], height_mm / data_smooth.shape[0]
    polys_to_merge = []
    candidates = []
    for contour in contours:
        coords = [( (p[1] - 1) * scale_x, height_mm - ((p[0] - 1) * scale_y) ) for p in contour]
        if len(coords) < 3: continue
        poly = Polygon(coords)
        if not poly.is_valid: poly = make_valid(poly)
        clipped = poly.intersection(frame_poly_shape)
        if clipped.is_empty: continue
        geoms = list(clipped.geoms) if clipped.geom_type == 'MultiPolygon' else [clipped]
        for g in geoms:
            candidates.append(g)
            # Enforce min_area strictly on the top layer. On lower layers, keep edge-touching pieces to maintain wall continuity.
            if g.area >= settings['min_area'] or (not is_top_layer and g.intersects(frame_poly_shape.boundary)): polys_to_merge.append(g)
    
    if not polys_to_merge: return []
    merged_geom = unary_union(polys_to_merge)

    geoms = list(merged_geom.geoms) if merged_geom.geom_type == 'MultiPolygon' else [merged_geom]
    return [{'poly': g, 'type': 'terrain', 'area': g.area} for g in geoms if not g.is_empty]

def _generate_frame_geometry(settings):
    if settings['frame_mode'] == "None": return []
    w, W, H = settings['frame_width'], settings['box_w'], settings['box_h']
    frame_parts = []
    if settings['frame_mode'] == "Full Perimeter":
        outer, inner = box(0, 0, W, H), box(w, w, W-w, H-w)
        frame_parts.append(outer.difference(inner))
    elif settings['frame_mode'] == "Custom Sides":
        sides = settings['frame_sides']
        if sides['top']: frame_parts.append(box(0, H-w, W, H))
        if sides['bottom']: frame_parts.append(box(0, 0, W, w))
        if sides['left']: frame_parts.append(box(0, 0, w, H))
        if sides['right']: frame_parts.append(box(W-w, 0, W, H))
    if not frame_parts: return []
    frame_geom = unary_union(frame_parts)
    geoms = list(frame_geom.geoms) if frame_geom.geom_type == 'MultiPolygon' else [frame_geom]
    return [{'poly': g, 'type': 'frame', 'area': g.area} for g in geoms if not g.is_empty]

def _apply_dowels_to_geom(geom, settings, layer_index, total_layers):
    if not settings.get('use_dowels') or geom.is_empty: return geom
    r = settings.get('dowel_diam', 3.0) / 2.0
    holes = [Point(d.get('x'), d.get('y')).buffer(r) for d in settings.get('dowel_data', []) if layer_index <= (total_layers - d.get('skip', 1))]
    if not holes: return geom
    return geom.difference(unary_union(holes))

def _classify_polygons(geom, settings):
    if geom.is_empty: return []
    frame_poly_shape = box(0, 0, settings['box_w'], settings['box_h'])
    geoms = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    final_polygons = [{'poly': g, 'type': 'island' if not g.intersects(frame_poly_shape.boundary) else 'safe', 'area': g.area} for g in geoms]
    final_polygons.sort(key=lambda x: x['area'], reverse=True)
    return final_polygons

def auto_bridge_islands(layer_geoms, thickness, smooth_data, settings):
    """
    Intelligent Automation:
    Finds islands that are very close to the mainland (or other safe polygons)
    and automatically creates bridges to them.
    Threshold: < 1.5 * bridge_thickness
    """
    if not layer_geoms: return layer_geoms
    
    # Separation
    islands = []
    mainland = []
    
    for p in layer_geoms:
        if p['type'] == 'island': islands.append(p)
        else: mainland.append(p)
        
    if not mainland or not islands: return layer_geoms
    
    mainland_union = unary_union([p['poly'] for p in mainland if not p['poly'].is_empty])
    if mainland_union.is_empty: return layer_geoms
    
    # Spatial Indexing (STRtree) is overkill for small N, but efficient. 
    # For now, simple loop is fine as N is usually < 50.
    
    max_dist = thickness * 1.5
    bridged_islands = []
    remaining_islands = []
    
    new_bridges = []
    
    for island in islands:
        # Check distance to mainland
        dist = island['poly'].distance(mainland_union)
        
        if dist < max_dist:
            # Create bridge!
            bridge = create_intelligent_bridge(island['poly'], mainland_union, smooth_data, settings, thickness)
            if bridge:
                new_bridges.append(bridge)
                bridged_islands.append(island)
            else:
                remaining_islands.append(island)
        else:
            remaining_islands.append(island)
            
    if not new_bridges: return layer_geoms
    
    # Merge new bridges into mainland
    # We treat the bridged islands as 'safe' effectively, or just merge them into mainland poly
    # To preserve object identity (if we wanted), we could change type to 'safe'.
    # But usually 'safe' implies connected to frame.
    
    # Simplest approach: Add bridged islands + bridges to mainland geometry
    # But we need to return list of dicts.
    
    final_list = list(mainland)
    
    for island in bridged_islands:
         # Convert island to safe since it's now bridged
         final_list.append({'poly': island['poly'], 'type': 'safe', 'area': island['area']})
         
    for b in new_bridges:
         # Tag as auto_bridge so we can count them in UI
         final_list.append({'poly': b, 'type': 'auto_bridge', 'area': b.area})
         
    final_list.extend(remaining_islands)
    
    # Re-sort
    final_list.sort(key=lambda x: x['area'], reverse=True)
    
    return final_list

    # Re-sort
    final_list.sort(key=lambda x: x['area'], reverse=True)
    
    return final_list

def fuse_close_polygons(geom, gap=0.5):
    """
    Merges touching parts (both separate islands and self-intersecting arms)
    using asymmetric morphological closing.
    
    Dilates slightly MORE than it erodes, ensuring that the created bridges
    survive the erosion step.
    Net effect: Global thickening of ~0.15mm (imperceptible) but strong welding.
    """
    if geom.is_empty: return geom
    
    # 1. Dilate to bridge gaps
    # Add small bias (0.15mm) to ensure bridge is robust
    dilate_amt = (gap / 2.0) + 0.15
    dilated = geom.buffer(dilate_amt, join_style=2, resolution=16)
    
    # 2. Erode back to restore original dimensions (mostly)
    # We erode slightly less than we dilated to keep the weld.
    erode_amt = -(gap / 2.0)
    fused = dilated.buffer(erode_amt, join_style=2, resolution=16)
    
    return fused

def terrain_aware_thicken(poly, smooth_data, layer_index, m_per_layer, min_elev, settings, layer_below_polys=None):
    """
    Refined "Restricted Terrain Expansion" Thicken Strategy.
    
    1. STRICT LOCALITY: Defines a hard "Limit Mask" around the thin feature. 
       No geometry is ever added outside this mask, preventing "non-adjacent" modifications.
    2. ADAPTIVE FILL: Iteratively lowers the slicing plane locally within the mask 
       until the feature is robust enough.
    3. ORGANIC BLEND: Uses morphological smoothing to erase sharp clip lines.
    """
    if poly is None or poly.is_empty: return poly
    
    req_width = settings['min_feature_width']
    
    # --- Helpers: Consistent Organic Smoothing ---
    def morph_close(p, r):
        # Dilate then Erode: Fills cracks/gaps. Safe for thin features.
        if p.is_empty: return p
        return p.buffer(r, join_style=1, resolution=64).buffer(-r, join_style=1, resolution=64)

    def morph_open(p, r):
        # Erode then Dilate: Rounds corners but CAN DESTROY thin features.
        if p.is_empty: return p
        return p.buffer(-r, join_style=1, resolution=64).buffer(r, join_style=1, resolution=64)

    def morph_smooth(p, r_close, r_open=None):
        # Full pipeline with independent control
        # r_close: Determines the size of the "meniscus" or fillet at junctions (Tangency)
        # r_open: Determines the roundness of convex corners (Shape Character)
        if r_open is None: r_open = r_close * 0.5
        return morph_open(morph_close(p, r_close), r_open)

    # --- 1. Input Smoothing (Safe Sanitation) ---
    # The input is THIN. We cannot use 'Open' (Erosion) or we might delete it.
    # We only use 'Close' (dilation) to fuse pixel-stepped edges.
    # We use a very small radius just to handle raster noise.
    poly = morph_close(poly, req_width * 0.1) 
    
    # removed: poly = poly.simplify(0.02, preserve_topology=True)
    
    # --- 2. Define ROI & Limit Mask ---
    search_radius = req_width * 1.5
    
    # High-res buffer for the mask
    limit_mask = poly.buffer(search_radius, join_style=1, resolution=64)
    
    # --- 3. Local Terrain Extraction ---
    h, w = smooth_data.shape
    scale_x = settings['box_w'] / w
    scale_y = settings['box_h'] / h
    
    minx, miny, maxx, maxy = limit_mask.bounds
    pad_px = 2
    c0 = max(0, int(minx / scale_x) - pad_px)
    c1 = min(w, int(maxx / scale_x) + pad_px)
    r0 = max(0, int((settings['box_h'] - maxy) / scale_y) - pad_px)
    r1 = min(h, int((settings['box_h'] - miny) / scale_y) + pad_px)
    
    if r1 <= r0 or c1 <= c0: 
        # Smooth Fallback (High Res)
        base = poly.buffer(req_width * 0.4, join_style=1, resolution=32)
        return morph_smooth(base, req_width * 0.3)
    
    local_data = smooth_data[r0:r1, c0:c1]
    
    # --- 4. Restricted Gradient Descent ---
    current_elev = min_elev + (layer_index - 1.0 + 0.5) * m_per_layer
    max_depth = m_per_layer * 1.5 
    n_steps = 10
    step_size = max_depth / n_steps
    
    best_patch = None
    original_area = poly.area
    
    fast_preview = settings.get('fast_preview', False)

    for i in range(n_steps):
        test_elev = current_elev - (i * step_size)
        padded_local = np.pad(local_data, pad_width=1, mode='constant', constant_values=test_elev - 1.0)
        contours = measure.find_contours(padded_local, test_elev)
        
        candidates = []
        for contour in contours:
            coords = []
            for p in contour:
                global_c = c0 + (p[1] - 1)
                global_r = r0 + (p[0] - 1)
                x = global_c * scale_x
                y = settings['box_h'] - (global_r * scale_y)
                coords.append((x, y))
            
            if len(coords) < 3: continue
            
            c_poly = Polygon(coords)
            if not c_poly.is_valid: c_poly = make_valid(c_poly)
            
            if c_poly.intersects(poly) or c_poly.distance(poly) < 0.1:
                clipped = c_poly.intersection(limit_mask)
                if not clipped.is_empty:
                    # Smooth the candidate piece
                    smoothed_cand = morph_smooth(clipped, 0.2)
                    candidates.append(smoothed_cand)
        
        if not candidates: continue
        
        merged_candidate = unary_union(candidates)
        
        # RELAXED CRITERIA:
        # Just needs to be bigger. The "1.1" (10%) growth was too strict for small fixes.
        if merged_candidate.area > original_area * 1.01:
            best_patch = merged_candidate
            eroded = merged_candidate.buffer(-req_width * 0.4, resolution=64)
            if not eroded.is_empty:
                break
                
    # --- 5. Assembly & Final Smoothing ---
    if best_patch:
        combined = unary_union([poly, best_patch])
        # Smooth the result organically
        # BALANCED BLEND: We dial back the radii to avoid "melting" the original shape.
        # Close: 0.8x (Good fillet, but respects concave features)
        # Open: 0.3x (Keeps definition, just softens the junction)
        final_poly = morph_smooth(combined, req_width * 0.8, req_width * 0.3)
    else:
        # Smooth Fallback
        base = poly.buffer(req_width * 0.4, join_style=1, resolution=64)
        # Apply balanced blend to fallback too
        final_poly = morph_smooth(base, req_width * 0.8, req_width * 0.3)

    # Final Check & Cleanup
    if final_poly.is_empty: return poly
    
    # NO final simplification. We want to keep the high-res 32-segment arcs.
    # But ensure we don't re-introduce angularity.
    # final_poly = final_poly.simplify(0.05, preserve_topology=True)

    # --- 5. Support Constraint (Gravity Check) ---
    if layer_below_polys:
        support_geom = unary_union(layer_below_polys)
        supported = final_poly.intersection(support_geom)
        
        # If we clipped it too much (e.g. overhanging into nothing), 
        # we might want to keep at least the original poly to avoid deleting the feature entirely.
        if supported.area < poly.area * 0.9:
             # Safety: return original or simple buffered original if supported is garbage
             # But better to return the supported part + original intersection?
             # Let's assume if it was originally there, it was supported (or user accepted it).
             # We just ensure the *extension* is supported.
             
             # Actually, simpler: Union(supported, poly.intersection(support_geom))
             # But poly might be floating if it's a problem feature.
             # Let's just return supported.
             return supported
             
        return supported

    return final_poly

def create_intelligent_bridge(poly1, poly2, smooth_data, settings, thickness):
    """
    Creates a more natural-looking bridge between two polygons by finding a path
    along the highest elevation points and creating a smooth, pill-shaped connection.
    """
    if smooth_data is None: return None

    width_mm, height_mm = settings['box_w'], settings['box_h']
    max_elev = np.max(smooth_data)
    cost = (max_elev - smooth_data) + 1

    p1, p2 = nearest_points(poly1, poly2)
    h, w = smooth_data.shape
    start_col, start_row = int(p1.x / width_mm * w), int((height_mm - p1.y) / height_mm * h)
    end_col, end_row = int(p2.x / width_mm * w), int((height_mm - p2.y) / height_mm * h)
    
    start_row, start_col = np.clip(start_row, 0, h - 1), np.clip(start_col, 0, w - 1)
    end_row, end_col = np.clip(end_row, 0, h - 1), np.clip(end_col, 0, w - 1)

    try:
        path, _ = route_through_array(cost, (start_row, start_col), (end_row, end_col), fully_connected=True)
        if not path: path = [(start_row, start_col), (end_row, end_col)]
    except (ValueError, IndexError):
        path = [(start_row, start_col), (end_row, end_col)]

    path_coords = [( (col / w * width_mm), height_mm - (row / h * height_mm) ) for row, col in path]
    
    if len(path_coords) < 2: return None
        
    bridge_line = LineString(path_coords)
    
    # Buffer the line to create the bridge polygon, with wider ends for a natural transition
    main_bridge = bridge_line.buffer(thickness / 2, cap_style='round')
    start_point = Point(path_coords[0])
    end_point = Point(path_coords[-1])
    start_buffer = start_point.buffer(thickness * 0.75) # Flared end
    end_buffer = end_point.buffer(thickness * 0.75)   # Flared end
    final_bridge = unary_union([main_bridge, start_buffer, end_buffer])
    
    return make_valid(final_bridge)

def _poly_to_svg_path(geom, y_flip_height=None):
    """Converts a Shapely geometry to an SVG path string, optionally flipping the Y coordinate."""
    if geom is None or geom.is_empty: return ""
    if geom.geom_type == 'MultiPolygon': return " ".join([_poly_to_svg_path(p, y_flip_height) for p in geom.geoms])
    if geom.geom_type == 'Polygon':
        path_data = ""
        
        def get_coords(coords_list):
            if y_flip_height is not None:
                return [(c[0], y_flip_height - c[1]) for c in coords_list]
            return coords_list

        # Exterior
        ext_coords = list(geom.exterior.coords)
        if not ext_coords: return ""
        
        transformed_coords = get_coords(ext_coords)
        path_data += f"M {transformed_coords[0][0]:.3f} {transformed_coords[0][1]:.3f} " + " ".join([f"L {x:.3f} {y:.3f}" for x, y in transformed_coords[1:]]) + " Z "
        
        # Interiors
        for interior in geom.interiors:
            int_coords = list(interior.coords)
            if not int_coords: continue
            
            transformed_int_coords = get_coords(int_coords)
            path_data += f"M {transformed_int_coords[0][0]:.3f} {transformed_int_coords[0][1]:.3f} " + " ".join([f"L {x:.3f} {y:.3f}" for x, y in transformed_int_coords[1:]]) + " Z "
        return path_data
    return ""

def generate_svg_string(polygons, width_mm, height_mm, fill_color="black", stroke_color="gray", add_background=True, fill_opacity=1.0):
    """
    Generates a static SVG string for a given layer's geometries, intended for export.
    Uses the `svgwrite` library to ensure valid and clean output.
    Supports `fill_color` as a dictionary mapping 'type' -> color for mixed content.
    """
    s = StringIO()
    dwg = svgwrite.Drawing(profile='tiny', size=(f"{width_mm}mm", f"{height_mm}mm"), viewBox=f"0 0 {width_mm} {height_mm}")
    
    if add_background:
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white', stroke='#ddd', stroke_width=0.5))

    def add_path_group(polys, color, opacity):
        path_data = " ".join([_poly_to_svg_path(p['poly'], y_flip_height=height_mm) for p in polys])
        if path_data:
            dwg.add(dwg.path(d=path_data, fill=color, stroke=stroke_color, stroke_width=0.2, fill_rule="evenodd", fill_opacity=opacity))

    if isinstance(fill_color, dict):
        # Group by type
        groups = {}
        for p in polygons:
            t = p.get('type', 'part') # Default to 'part' if missing
            if t not in groups: groups[t] = []
            groups[t].append(p)
        
        # Render Order: Parts then Jigs (or specific order)
        # We render unknown types first, then 'part', then 'jig' on top?
        # Actually standard dict order is insert order in Py3.7+, but let's be explicit if needed.
        # Just iterate.
        for t, group in groups.items():
            f_col = fill_color.get(t, fill_color.get('default', 'black'))
            # Special case: Jigs usually translucent?
            # We can use fill_opacity dict if we wanted, but let's stick to global opacity for now, 
            # OR pass opacity in the color map? No, color map is just strings.
            # Let's assume Jigs might need lower opacity.
            cur_opacity = 0.8 if t == 'jig' else fill_opacity
            add_path_group(group, f_col, cur_opacity)
            
    else:
        # Legacy Single Color
        add_path_group(polygons, fill_color, fill_opacity)

    dwg.write(s)
    return s.getvalue()

def filter_geometries_based_on_decisions(geoms, decisions):
    filtered = []
    for i, layer in enumerate(geoms):
        islands = [p for p in layer if p['type'] == 'island']
        if not islands:
            filtered.append(layer)
            continue
        new_layer = [p for p in layer if p['type'] != 'island']
        for j, island in enumerate(islands):
            if not decisions.get(f"L{i+1}_I{j}", False): new_layer.append(island)
        filtered.append(new_layer)
    return filtered

def auto_heal_layer(raw_layer_geoms, smooth_data, layer_index, m_per_layer, min_elev, settings, layer_below_polys=None):
    """
    SMART AUTO-HEAL:
    1. Analyzes the layer for thin features.
    2. Filters out "noise" (tiny specs).
    3. Automatically THICKENS valid thin features using Restricted Terrain Expansion.
    4. Returns the healed layer AND a list of modification polygons (for "Trust but Verify").
    """
    min_width = settings.get('min_feature_width', 3.0)
    
    # We process the UNION of the raw layer to find problems globally first
    # But usually raw_layer_geoms is a list of dicts.
    # We need to process each connected component? 
    # Actually, analyze_thin_features works on a single geometry.
    # So we should Union -> Analyze -> Thicken -> Re-Separate?
    
    # Better approach:
    # 1. Union all 'island' and 'safe' (everything except frame?)
    # 2. Analyze.
    # 3. For every "bridge" (problem) found:
    #    - Thicken it.
    #    - Add it to the geometry.
    
    current_polys = [p['poly'] for p in raw_layer_geoms if not p['poly'].is_empty and p['type'] in ['island', 'safe', 'terrain']]
    if not current_polys: return raw_layer_geoms, []
    
    combined_raw = unary_union(current_polys)
    
    # --- AUTO-FUSE (Close small gaps) ---
    if settings.get('auto_fuse', True):
        fuse_gap = settings.get('fuse_gap', 1.0)
        if fuse_gap > 0:
            # Buffer out then in to bridge gaps
            # Use 'join_style=2' (mitre) or '1' (round)? Round is safer for organic.
            combined_raw = combined_raw.buffer(fuse_gap / 2.0, join_style=1).buffer(-(fuse_gap / 2.0), join_style=1)
    
    # Clean & Analyze
    # Note: 'analyze_thin_features' returns (cleaned_base, problem_list)
    # where cleaned_base is the geometry WITHOUT the thin parts.
    robust_base, problems = analyze_thin_features(combined_raw, min_width, auto_cleanup=True)
    
    # If no problems, we just return the original list (or re-classified robust base?)
    # Using robust_base might alter the original shape slightly (erosion/dilation).
    # To be "Trust but Verify", we should strictly ADD to the original.
    
    # Wait, if we keep the original, we keep the thin parts.
    # Then we Thicken the thin parts and Union them back.
    # So: Final = Original U (Thickened Problems)
    
    modifications = []
    
    final_geom = combined_raw
    
    if problems:
        for prob in problems:
            # NOISE FILTER (Aggressive)
            # If a problem is tiny and disconnected, it's noise.
            # analyze_thin_features already does some, but let's be strict.
            # UPDATED: Lowered threshold (0.05) to ensure thin bridges (merges) are not discarded.
            if prob.area < 0.05: continue
            
            # Thicken it!
            thickened = terrain_aware_thicken(prob, smooth_data, layer_index, m_per_layer, min_elev, settings, layer_below_polys)
            
            # Calculate what we ADDED (for visualization)
            # Use make_valid to handle potentially tricky boolean results
            raw_diff = thickened.difference(combined_raw)
            added_mass = make_valid(raw_diff)
            
            if not added_mass.is_empty:
                modifications.append(added_mass)
            
            # Add to final
            final_geom = final_geom.union(thickened)
            
    # Re-classify into Island/Safe list
    new_layer_list = _classify_polygons(final_geom, settings)
    
    # FINAL DE-SPECKLE: Explicitly remove tiny noise
    noise_threshold = (min_width * min_width) * 0.2
    new_layer_list = [p for p in new_layer_list if p['poly'].area >= noise_threshold]
    
    # Ensure modifications are valid polygons for UI
    valid_modifications = []
    for m in modifications:
        geoms = list(m.geoms) if m.geom_type == 'MultiPolygon' else [m]
        for g in geoms:
             if g.area > 0.01:
                 # Assign ID for persistence
                 cx, cy = int(g.centroid.x * 10), int(g.centroid.y * 10)
                 valid_modifications.append({'poly': g, 'type': 'modification', 'area': g.area, 'uid': f"mod_{cx}_{cy}"})

    return new_layer_list, valid_modifications

def apply_merges_to_geometries(geoms, original_geoms_all_layers, groups, thickness, smooth_data, settings):
    processed = copy.deepcopy(geoms)
    for layer_key, group_list in groups.items():
        try: layer_idx = int(layer_key.lstrip('L')) - 1
        except ValueError: continue
        if layer_idx <= 0 or layer_idx >= len(processed): continue
        
        original_layer = original_geoms_all_layers[layer_idx]
        original_islands = [p for p in original_layer if p['type'] == 'island']

        current_layer = processed[layer_idx]
        current_islands = [p for p in current_layer if p['type'] == 'island']
        polys_to_keep = [p for p in current_layer if p['type'] != 'island']
        
        all_merged_original_indices = {idx for group in group_list for idx in group}
        
        for group in group_list:
            islands_to_merge_original = [original_islands[i] for i in group if i < len(original_islands)]
            islands_to_merge_wkt = {p['poly'].wkt for p in islands_to_merge_original}
            geoms_to_merge = [p for p in current_islands if p['poly'].wkt in islands_to_merge_wkt]
            
            if len(geoms_to_merge) < 2: continue

            geoms_to_merge_polys = [p['poly'] for p in geoms_to_merge]
            bridges = [create_intelligent_bridge(geoms_to_merge_polys[i], geoms_to_merge_polys[i+1], smooth_data, settings, thickness) for i in range(len(geoms_to_merge_polys) - 1)]
            final_poly = unary_union(geoms_to_merge_polys + [b for b in bridges if b])
            polys_to_keep.append({'poly': final_poly, 'type': 'safe', 'area': final_poly.area})

        unmerged_islands_wkt = {original_islands[i]['poly'].wkt for i in range(len(original_islands)) if i not in all_merged_original_indices}
        for island in current_islands:
            if island['poly'].wkt in unmerged_islands_wkt:
                polys_to_keep.append(island)

        processed[layer_idx] = polys_to_keep
    return processed

def apply_bridges_to_mainland(geoms, original_geoms_all_layers, requests, thickness, smooth_data, settings):
    processed = copy.deepcopy(geoms)
    for layer_key, indices in requests.items():
        if not indices: continue
        try: layer_idx = int(layer_key.lstrip('L')) - 1
        except ValueError: continue
        if layer_idx <= 0: continue

        original_layer = original_geoms_all_layers[layer_idx]
        original_islands = [p for p in original_layer if p['type'] == 'island']

        current_layer = processed[layer_idx]
        safe_polys = [p for p in current_layer if p['type'] == 'safe']
        if not safe_polys: continue
        mainland = max(safe_polys, key=lambda p: p['area'])
        
        current_islands = [p for p in current_layer if p['type'] == 'island']
        islands_to_bridge_original = [original_islands[i] for i in indices if i < len(original_islands)]
        islands_to_bridge_wkt = {p['poly'].wkt for p in islands_to_bridge_original}
        islands_to_bridge = [p for p in current_islands if p['poly'].wkt in islands_to_bridge_wkt]

        if not islands_to_bridge: continue

        union_geoms = [mainland['poly']] + [i['poly'] for i in islands_to_bridge]
        bridges = [create_intelligent_bridge(i['poly'], mainland['poly'], smooth_data, settings, thickness) for i in islands_to_bridge]
        final_union = unary_union(union_geoms + [b for b in bridges if b])
        
        other_polys = [p for p in current_layer if p != mainland and p not in islands_to_bridge]
        other_polys.append({'poly': final_union, 'type': 'safe', 'area': final_union.area})
        processed[layer_idx] = other_polys
    return processed

def apply_manual_bridges(geoms, points, thickness, settings, smooth_data):
    processed = copy.deepcopy(geoms)
    bridge_status = {} # Map layer_key -> {index -> status_string}

    # Sort keys to ensure bottom-up processing so bridges on lower layers exist for upper layers
    sorted_keys = sorted(points.keys(), key=lambda k: int(k.lstrip('L')))
    
    for layer_key in sorted_keys:
        point_list = points[layer_key]
        bridge_status[layer_key] = {}
        if not point_list: continue
        try: layer_idx = int(layer_key.lstrip('L')) - 1
        except ValueError: continue
        if layer_idx <= 0: continue

        current_layer = processed[layer_idx]
        polygons_on_layer = [p['poly'] for p in current_layer]
        
        # Get layer below geometry for constraint
        layer_below = processed[layer_idx - 1]
        layer_below_geom = unary_union([p['poly'] for p in layer_below])
        support_geom = layer_below_geom
        
        for i, point in enumerate(point_list):
            # Apply perturbation if present (for regeneration)
            px = point['x'] + point.get('dx', 0.0)
            py = point['y'] + point.get('dy', 0.0)
            click = Point(px, py)
            
            # 0. Validate click is on support
            if not click.within(support_geom):
                # Try to snap to nearest support within reasonable distance (e.g. 20mm)
                _, nearest_support_pt = nearest_points(click, support_geom)
                if click.distance(nearest_support_pt) < 20:
                    click = nearest_support_pt
                else:
                    bridge_status[layer_key][i] = "Invalid: Not on layer below"
                    continue

            # Strategy: Find the two closest distinct polygons to the click point
            valid_polys = [p for p in polygons_on_layer if not p.is_empty]
            if len(valid_polys) < 2:
                bridge_status[layer_key][i] = "Invalid: <2 islands nearby"
                continue
            
            # Sort by distance to click
            polys_with_dist = sorted([(p.distance(click), p) for p in valid_polys], key=lambda x: x[0])
            
            # Helper to verify connectivity
            def is_connected(geom, target_pt):
                if geom.is_empty: return False
                parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
                return any(p.distance(target_pt) < 0.2 for p in parts)

            # --- Strategy A: Try Direct Connection First (Smart Auto-Bridge) ---
            p1, p2 = polys_with_dist[0][1], polys_with_dist[1][1]
            p1_near, p2_near = nearest_points(p1, p2)
            direct_line = LineString([p1_near, p2_near])
            
            if direct_line.difference(support_geom).length < 0.1 and direct_line.distance(click) < 30:
                bridge = create_intelligent_bridge(p1_near.buffer(0.1), p2_near.buffer(0.1), smooth_data, settings, thickness)
                if bridge and not bridge.is_empty:
                    constrained = bridge.intersection(layer_below_geom)
                    if is_connected(constrained, p1_near) and is_connected(constrained, p2_near):
                        polygons_on_layer.append(constrained)
                        bridge_status[layer_key][i] = "Success"
                        continue 

            # --- Strategy B: Routed Connection (Through Click) ---
            def get_best_point(poly, target, support):
                _, near_pt = nearest_points(target, poly)
                line = LineString([near_pt, target])
                if line.difference(support).length < 0.1: return near_pt
                
                best_pt = near_pt
                min_dist = float('inf')
                sub_polys = poly.geoms if hasattr(poly, 'geoms') else [poly]
                for sub in sub_polys:
                    boundary = sub.exterior
                    if not boundary: continue
                    length = boundary.length
                    num_samples = max(20, int(length))
                    for k in range(num_samples):
                        pt = boundary.interpolate(k * length / num_samples)
                        line = LineString([pt, target])
                        if line.difference(support).length < 0.1:
                            d = pt.distance(target)
                            if d < min_dist:
                                min_dist = d
                                best_pt = pt
                return best_pt if min_dist != float('inf') else None
            
            connections = []
            for dist, poly in polys_with_dist:
                if dist > 100: continue
                pt = get_best_point(poly, click, support_geom)
                if pt: connections.append(pt)
                if len(connections) >= 2: break
            
            if len(connections) < 2:
                bridge_status[layer_key][i] = "Invalid: No valid path to >1 island"
                continue
            
            pt1, pt2 = connections[0], connections[1]
            b1 = create_intelligent_bridge(pt1.buffer(0.1), click.buffer(0.1), smooth_data, settings, thickness)
            b2 = create_intelligent_bridge(pt2.buffer(0.1), click.buffer(0.1), smooth_data, settings, thickness)
            bridge = unary_union([b for b in [b1, b2] if b])

            if bridge and not bridge.is_empty:
                constrained_bridge = bridge.intersection(layer_below_geom)
                if is_connected(constrained_bridge, click):
                    polygons_on_layer.append(constrained_bridge)
                    bridge_status[layer_key][i] = "Success"
                else:
                    bridge_status[layer_key][i] = "Invalid: Bridge severed by gap"
            else:
                bridge_status[layer_key][i] = "Failed to generate"
        
        processed[layer_idx] = _classify_polygons(unary_union(polygons_on_layer), settings)
    return processed, bridge_status

def _text_to_shapely(text, x, y, size):
    fp = FontProperties(family='sans-serif', weight='bold')
    tp = TextPath((x, y), text, size=size, prop=fp)
    polys = []
    for vertices in tp.to_polygons():
        if len(vertices) > 2: polys.append(Polygon(vertices))
    return unary_union(polys)

def _old_generate_jig_geometry(layer_geoms, dowels, box_w, box_h, layer_num, modifications=None, offset=6.0, conn_width=6.0, anchor_thick=6.0, anchor_len=40.0):
    all_layer_polys = []
    targets = []
    stable_polys = []
    edges = [(LineString([(0,0), (0, box_h)]), box_h), (LineString([(box_w,0), (box_w, box_h)]), box_h), (LineString([(0,0), (box_w, 0)]), box_w), (LineString([(0, box_h), (box_w, box_h)]), box_w)]

    for p in layer_geoms:
        if 'poly' in p and not p['poly'].is_empty:
            poly = p['poly']
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_empty: continue
            all_layer_polys.append(poly)
            
            # Check specific edge contacts to determine stability (must lock into a corner)
            # Edges: 0:Left, 1:Right, 2:Bottom, 3:Top
            touches = [poly.intersection(e[0]).length >= (e[1] * 0.05) for e in edges]
            
            # Stable if touching adjacent edges (Corner lock)
            is_stable = (touches[0] and touches[3]) or \
                        (touches[3] and touches[1]) or \
                        (touches[1] and touches[2]) or \
                        (touches[2] and touches[0])
            
            if not is_stable: targets.append(poly)
            else: stable_polys.append(poly)
            
    if not targets: return None
    
    connected_group = list(stable_polys)
    corner_brackets = []
    corners = [(0,0), (box_w, 0), (box_w, box_h), (0, box_h)]
    
    def is_corner_clear(cx, cy):
        dx = 1 if cx == 0 else -1
        dy = 1 if cy == 0 else -1
        b1 = box(min(cx, cx + anchor_len*dx), min(cy, cy + anchor_thick*dy), max(cx, cx + anchor_len*dx), max(cy, cy + anchor_thick*dy))
        b2 = box(min(cx, cx + anchor_thick*dx), min(cy, cy + anchor_len*dy), max(cx, cx + anchor_thick*dx), max(cy, cy + anchor_len*dy))
        check_box = unary_union([b1, b2])
        for obs in all_layer_polys:
            if check_box.intersects(obs): return False
        return True

    valid_corners = [c for c in corners if is_corner_clear(c[0], c[1])]
    for cx, cy in valid_corners:
        dx = 1 if cx == 0 else -1
        dy = 1 if cy == 0 else -1
        b1 = box(min(cx, cx + anchor_len*dx), min(cy, cy + anchor_thick*dy), max(cx, cx + anchor_len*dx), max(cy, cy + anchor_thick*dy))
        b2 = box(min(cx, cx + anchor_thick*dx), min(cy, cy + anchor_len*dy), max(cx, cx + anchor_thick*dx), max(cy, cy + anchor_len*dy))
        bracket = unary_union([b1, b2])
        corner_brackets.append(bracket)
        connected_group.append(bracket)

    if not connected_group:
        best_corner = None
        min_overlap = float('inf')
        for c in corners:
            cx, cy = c
            dx = 1 if cx == 0 else -1
            dy = 1 if cy == 0 else -1
            b1 = box(min(cx, cx + anchor_len*dx), min(cy, cy + anchor_thick*dy), max(cx, cx + anchor_len*dx), max(cy, cy + anchor_thick*dy))
            b2 = box(min(cx, cx + anchor_thick*dx), min(cy, cy + anchor_len*dy), max(cx, cx + anchor_thick*dx), max(cy, cy + anchor_len*dy))
            bracket = unary_union([b1, b2])
            overlap_area = sum(bracket.intersection(p).area for p in all_layer_polys)
            if overlap_area < min_overlap:
                min_overlap = overlap_area
                best_corner = c
        cx, cy = best_corner
        dx = 1 if cx == 0 else -1; dy = 1 if cy == 0 else -1
        b1 = box(min(cx, cx + anchor_len*dx), min(cy, cy + anchor_thick*dy), max(cx, cx + anchor_len*dx), max(cy, cy + anchor_thick*dy))
        b2 = box(min(cx, cx + anchor_thick*dx), min(cy, cy + anchor_len*dy), max(cx, cx + anchor_thick*dx), max(cy, cy + anchor_len*dy))
        bracket = unary_union([b1, b2])
        corner_brackets.append(bracket)
        connected_group.append(bracket)

    unconnected_group = list(targets)
    connections = []
    used_anchors = set()
    safe_dist = (conn_width / 2.0)
    all_obstacles = all_layer_polys

    def extend_endpoints(p_start, p_end, dist):
        dx = p_end.x - p_start.x
        dy = p_end.y - p_start.y
        length = math.hypot(dx, dy)
        if length == 0: return p_start, p_end
        ux, uy = dx / length, dy / length
        return Point(p_start.x - ux*dist, p_start.y - uy*dist), Point(p_end.x + ux*dist, p_end.y + uy*dist)

    while unconnected_group:
        best_dist = float('inf')
        best_pair = (None, None, None)
        best_mid = None
        for i, island in enumerate(unconnected_group):
            for target in connected_group:
                p1, p2 = nearest_points(island, target)
                dist = p1.distance(p2)
                if dist < best_dist:
                    line = LineString([p1, p2])
                    intersects = False
                    if line.length > 0.2:
                        for obs in all_obstacles:
                            if obs is island: continue
                            if isinstance(target, (Polygon, MultiPolygon)) and obs is target: continue
                            if line.distance(obs) < safe_dist:
                                intersects = True; break
                    if not intersects:
                        best_dist = dist
                        best_pair = (i, (p1, p2, 'direct'), target)
                        best_mid = None
                    else:
                        dx = p2.x - p1.x; dy = p2.y - p1.y
                        mid_x, mid_y = (p1.x + p2.x)/2, (p1.y + p2.y)/2
                        len_v = math.hypot(dx, dy)
                        if len_v > 0:
                            nx, ny = -dy/len_v, dx/len_v
                            found_dogleg = False
                            for mag_factor in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
                                offset_mag = min(dist * mag_factor, box_w/2)
                                raw_candidates = [Point(mid_x + nx*offset_mag, mid_y + ny*offset_mag), Point(mid_x - nx*offset_mag, mid_y - ny*offset_mag)]
                                candidates = []
                                margin = 2.0
                                for c in raw_candidates:
                                    cx = min(max(c.x, margin), box_w - margin)
                                    cy = min(max(c.y, margin), box_h - margin)
                                    candidates.append(Point(cx, cy))
                                for mid_pt in candidates:
                                    l1 = LineString([p1, mid_pt]); l2 = LineString([mid_pt, p2])
                                    path_len = l1.length + l2.length
                                    if path_len < best_dist:
                                        valid_dogleg = True
                                        for seg in [l1, l2]:
                                            if seg.length > 0.2:
                                                for obs in all_obstacles:
                                                    if obs is island: continue
                                                    if isinstance(target, (Polygon, MultiPolygon)) and obs is target: continue
                                                    if seg.distance(obs) < safe_dist:
                                                        valid_dogleg = False; break
                                            if not valid_dogleg: break
                                        if valid_dogleg:
                                            best_dist = path_len
                                            best_pair = (i, (p1, p2, 'dogleg'), target)
                                            best_mid = mid_pt
                                            found_dogleg = True; break
                                if found_dogleg: break
        if best_pair[0] is not None:
            idx, (pt_island, pt_target, p_type), target_obj = best_pair
            if target_obj in corner_brackets: used_anchors.add(target_obj)
            if p_type == 'direct':
                p1_ext, p2_ext = extend_endpoints(pt_island, pt_target, conn_width)
                connections.append(LineString([p1_ext, p2_ext]))
            else:
                p1_ext, _ = extend_endpoints(pt_island, best_mid, conn_width)
                _, p2_ext = extend_endpoints(best_mid, pt_target, conn_width)
                connections.append(LineString([p1_ext, best_mid])); connections.append(LineString([best_mid, p2_ext]))
            connected_group.append(unconnected_group.pop(idx))
        else: break

    targets_buffered = unary_union([p.buffer(offset) for p in targets])
    connections_buffered = unary_union([l.buffer(conn_width / 2.0) for l in connections])
    anchors_buffered = unary_union(list(used_anchors))
    raw_jig = unary_union([targets_buffered, anchors_buffered, connections_buffered])
    box_poly = box(0, 0, box_w, box_h)
    jig_clipped = raw_jig.intersection(box_poly)
    cleaned_jig = jig_clipped.buffer(-0.1).buffer(0.1)
    modified_jig = cleaned_jig
    if modifications:
        for mod in modifications:
            mx, my, mw, mh = mod['x'], mod['y'], mod['w'], mod['h']
            shape = box(mx - mw/2, my - mh/2, mx + mw/2, my + mh/2)
            if mod['type'] == 'add':
                shape = shape.intersection(box_poly)
                if not shape.is_empty and shape.intersects(modified_jig): modified_jig = modified_jig.union(shape)
            elif mod['type'] == 'sub': modified_jig = modified_jig.difference(shape)

    component_footprints = []
    clearance = 0.2
    for p in all_layer_polys:
        if p.is_empty: continue
        polys = [p] if p.geom_type == 'Polygon' else list(p.geoms)
        for sub in polys:
            filled = Polygon(sub.exterior)
            component_footprints.append(filled)
    holes_with_clearance = unary_union([p.buffer(clearance) for p in component_footprints])
    jig = modified_jig.difference(holes_with_clearance)
    dowel_points = [Point(d['x'], d['y']) for d in dowels]
    if dowel_points: jig = jig.difference(unary_union([d.buffer(1.6) for d in dowel_points]))
    if not jig.is_valid: jig = jig.buffer(0)
    jig = jig.intersection(box_poly)
    
    # Final Cleanup: Remove thin parts (< 1.5mm) to prevent burning/breaking
    jig, _ = clean_geometry(jig, min_width=1.5)
    
    jig = jig.intersection(box_poly)
    if jig.is_empty: return None
    if jig.geom_type == 'GeometryCollection':
        polys = [g for g in jig.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        jig = unary_union(polys)
    elif jig.geom_type not in ['Polygon', 'MultiPolygon']: return None
    if jig.geom_type == 'MultiPolygon':
        ground_anchors = unary_union(list(used_anchors)) if used_anchors else Polygon()
        ground_stable = unary_union(stable_polys) if stable_polys else Polygon()
        valid_parts = []
        for part in jig.geoms:
            is_anchored = (not ground_anchors.is_empty and part.intersects(ground_anchors))
            is_stable = (not ground_stable.is_empty and part.distance(ground_stable) < 0.5)
            if is_anchored or is_stable: valid_parts.append(part)
        if valid_parts: jig = unary_union(valid_parts)
        else: jig = max(jig.geoms, key=lambda p: p.area)
    return {'poly': jig, 'labels': []}

def remove_small_islands(geom, min_area=50.0):
    """Retains only polygons larger than min_area."""
    if geom is None or geom.is_empty: return geom
    if geom.geom_type == 'Polygon':
        return geom if geom.area >= min_area else Polygon()
    elif geom.geom_type == 'MultiPolygon':
        valid = [p for p in geom.geoms if p.area >= min_area]
        return unary_union(valid)
    return geom

def generate_jig_geometry(layer_geoms, _dowels, box_w, box_h, layer_num, modifications=None, offset=6.0, conn_width=6.0, anchor_thick=6.0, anchor_len=40.0, grid_spacing=20.0, fluid_smoothing=True):
    """
    Generates a "Unified Spider" Jig.
    
    ALGORITHM: Unified Orthogonal Anchors.
    Goal: A SINGLE, rigid tool per layer that positions all floating parts.
    Logic:
    1. Generate Orthogonal Anchors (Spider) for stability.
    2. Filter out tiny debris (useless bits).
    3. FORCE CONNECTIVITY: Bridge any disjoint components to form one single tool.
       - This ensures the user doesn't manage 20 small clips.
    """
    if not layer_geoms: return None
    
    islands = []
    for p in layer_geoms:
        if 'poly' in p and not p['poly'].is_empty:
            poly = p['poly']
            if not poly.is_valid: poly = poly.buffer(0)
            if not poly.is_empty: islands.append(poly)
            
    if not islands: return None
    
    CLEARANCE = 0.25 
    
    # Zones
    parts_union = unary_union(islands)
    holes_mask = parts_union.buffer(CLEARANCE, join_style=2)
    
    # Collars
    # --- PIVOT v3.1: SMART GRID JIG ---
    # Constraint-Aware: Only support Active (Floating) parts.
    
    # 0. Constraint Analysis
    active_indices = []
    active_geoms = []
    
    for i, island in enumerate(islands):
        b_minx, b_miny, b_maxx, b_maxy = island.bounds
        
        # Corner Lock Rule: Touch X (L/R) AND Y (T/B)
        touch_x = (b_minx <= 1.0) or (b_maxx >= box_w - 1.0)
        touch_y = (b_miny <= 1.0) or (b_maxy >= box_h - 1.0)
        
        if touch_x and touch_y:
            # Fixed. No Jig needed.
            continue
        else:
            active_indices.append(i)
            active_geoms.append(island)
            
    if not active_indices:
        return None
        
    active_union = unary_union(active_geoms)
    # active_union needs a slight buffer to ensure intersection checks work reliably
    active_check_mask = active_union.buffer(1.0)

    # 1. Generate Collars (Only for Active Parts)
    collar_shapes = []
    for i in active_indices:
        island = islands[i]
        collar = island.buffer(CLEARANCE + conn_width, join_style=2)
        collar_shapes.append(collar)
    
    # 2. Virtual Grid Rays
    grid_spacing = 40.0
    valid_struts = []
    
    # All obstacles (Parts + Hole Environments?)
    # We want to stop at parts.
    obstacles_geom = parts_union
    
    # Define Scan Lines
    scan_lines = []
    
    # 1. Standard Grid
    x = grid_spacing
    while x < box_w:
        scan_lines.append(LineString([(x, -10), (x, box_h+10)]))
        x += grid_spacing
    y = grid_spacing
    while y < box_h:
        scan_lines.append(LineString([(-10, y), (box_w+10, y)]))
        y += grid_spacing
        
    # 2. Grid Insurance (Centroid Scans)
    # Ensure every active part is hit by at least one vertical and one horizontal line.
    for i in active_indices:
        c = islands[i].centroid
        scan_lines.append(LineString([(c.x, -10), (c.x, box_h+10)]))
        scan_lines.append(LineString([(-10, c.y), (box_w+10, c.y)]))
        
    # 3. Process Scans
    # Deduplicate/Filter Scan Lines close to each other
    # We have vertical lines (constant X) and horizontal (constant Y).
    # We need to sort and filter them.
    
    # Separate V and H
    v_lines = sorted([l for l in scan_lines if l.coords[0][0] == l.coords[1][0]], key=lambda l: l.coords[0][0])
    h_lines = sorted([l for l in scan_lines if l.coords[0][1] == l.coords[1][1]], key=lambda l: l.coords[0][1])
    
    unique_scan_lines = []
    
    MIN_LINE_SPACING = 15.0
    
    # Filter Vertical
    last_pos = -999.0
    for l in v_lines:
        pos = l.coords[0][0]
        if pos - last_pos >= MIN_LINE_SPACING:
            unique_scan_lines.append(l)
            last_pos = pos
            
    # Filter Horizontal
    last_pos = -999.0
    for l in h_lines:
        pos = l.coords[0][1]
        if pos - last_pos >= MIN_LINE_SPACING:
            unique_scan_lines.append(l)
            last_pos = pos
            
    for line in unique_scan_lines:
        # ... (Intersection Logic) ...
        try:
            inter = line.intersection(parts_union)
        except: continue
        if inter.is_empty: continue
            
        gaps = line.difference(parts_union.buffer(0.5))
        
        gap_segs = []
        if gaps.geom_type == 'LineString': gap_segs.append(gaps)
        elif gaps.geom_type == 'MultiLineString': gap_segs.extend(gaps.geoms)
            
        for gap in gap_segs:
            # Revert to Robust Selection (v3.4 style): Keep everything initially.
            if gap.intersects(active_check_mask):
                valid_struts.append(gap)
            
    # 3.5. CONSTRAINT ENFORCEMENT (The "Safety Net" - v3.4 Logic)
    # Ensure EVERY active island has at least one Vertical and one Horizontal valid connection.
    for i in active_indices:
        island = islands[i]
        c = island.centroid
        
        # Check current connectivity
        has_v = False
        has_h = False
        
        check_zone = island.buffer(0.5)
        
        for strut in valid_struts:
            if not strut.intersects(check_zone): continue
            
            p1 = Point(strut.coords[0])
            p2 = Point(strut.coords[-1])
            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)
            
            if dy > dx: has_v = True
            else: has_h = True
            
            if has_v and has_h: break
            
        # Force Add if missing
        missing_axes = []
        if not has_v: missing_axes.append('V')
        if not has_h: missing_axes.append('H')
        
        for axis in missing_axes:
            if axis == 'V': line = LineString([(c.x, -10), (c.x, box_h+10)])
            else: line = LineString([(-10, c.y), (box_w+10, c.y)])
            
            try:
                gaps = line.difference(parts_union.buffer(0.5))
            except: continue
            
            gap_segs = []
            if gaps.geom_type == 'LineString': gap_segs.append(gaps)
            elif gaps.geom_type == 'MultiLineString': gap_segs.extend(gaps.geoms)
            
            for gap in gap_segs:
                if gap.intersects(check_zone):
                     valid_struts.append(gap)
                     break
                     
    # 3.6. SAFE OPTIMIZATION (Prune Redundant Long Struts)
    # User Request: "Optimize material use... spanning full width" implies dropping long tails.
    # Logic: If a part has a SHORT strut on an axis, drop the LONG struts on that same axis.
    # Safety: ONLY drop if a standard 'good' strut exists.
    
    final_struts = []
    # Identify which struts serve which active part (Index Mapping)
    # A strut can serve multiple parts (bridge).
    # We want to keep a strut if it is critical for ANY part it touches.
    
    # Map Strut Index -> List of Active Part Indices it touches
    strut_to_parts = {}
    for si, strut in enumerate(valid_struts):
        strut_to_parts[si] = []
        for ai in active_indices:
            # Reuse check logic
            if strut.distance(islands[ai]) < 1.0:
                 strut_to_parts[si].append(ai)
                 
    # Map Part Index -> List of Strut Indices attached to it
    part_to_struts = {}
    for ai in active_indices:
        part_to_struts[ai] = []
        
    for si, ais in strut_to_parts.items():
        for ai in ais:
            part_to_struts[ai].append(si)
            
    # Evaluation: Determine if a strut is "Redundant" for a specific part.
    # A strut is redundant for Part P if:
    # 1. Strut is Long (> 150mm).
    # 2. Part P has another Strut S2 on same axis that is Short (< 50mm).
    
    # We mark struts as "Necessary". A strut is necessary if it is needed by AT LEAST ONE part.
    # If a strut is redundant for Part A but necessary for Part B (e.g. bridge), it stays.
    necessary_indices = set()
    
    for ai in active_indices:
        my_strut_indices = part_to_struts[ai]
        
        # Split V/H
        v_sids = []
        h_sids = []
        
        for si in my_strut_indices:
            geom = valid_struts[si]
            p1, p2 = Point(geom.coords[0]), Point(geom.coords[-1])
            dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
            if dy > dx: v_sids.append(si)
            else: h_sids.append(si)
            
        # Analyze V
        if v_sids:
            # Find shortest V strut
            v_struts = [(si, valid_struts[si].length) for si in v_sids]
            v_struts.sort(key=lambda x: x[1]) # Shortest first
            
            best_si, best_len = v_struts[0]
            necessary_indices.add(best_si) # Always keep shortest
            
            # Check others against shortest
            for si, length in v_struts[1:]:
                # Relative Pruning Rule:
                # If Best is "Good Anchor" (< 100mm) AND This is "Significantly Worse" (> 1.5x), Prune.
                if best_len < 100.0 and length > 100.0 and length > (best_len * 1.5):
                     pass # Prune
                else:
                     necessary_indices.add(si)
        
        # Analyze H
        if h_sids:
             h_struts = [(si, valid_struts[si].length) for si in h_sids]
             h_struts.sort(key=lambda x: x[1])
             
             best_si, best_len = h_struts[0]
             necessary_indices.add(best_si)
             
             for si, length in h_struts[1:]:
                 if best_len < 100.0 and length > 100.0 and length > (best_len * 1.5):
                     pass
                 else:
                     necessary_indices.add(si)
                
    final_struts = [valid_struts[i] for i in sorted(list(necessary_indices))]
    
    # Generate Orientation Arrows (North-Pointing Triangles)
    # User Request: "Small triangles... pointed towards the top... easier to identify orientation."
    ARROW_BASE = 15.0 # Wider (4.5mm stickout per side)
    ARROW_HEIGHT = 12.0 # Taller 
    orientation_markers = []
    
    for strut in final_struts:
        # We place marker at midpoint
        p1, p2 = strut.coords[0], strut.coords[-1]
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        
        # North Pointing Triangle
        # Tip at (mid_x, mid_y + ARROW_HEIGHT/2)
        # Base Left at (mid_x - ARROW_BASE/2, mid_y - ARROW_HEIGHT/2)
        # Base Right at (mid_x + ARROW_BASE/2, mid_y - ARROW_HEIGHT/2)
        
        # Adjust Y to center the triangle on the strut's axis?
        # If strut is H, midpoint is on strut center.
        # If strut is V, midpoint is on strut center.
        # A symmetrical Triangle centered on (mid_x, mid_y) works.
        
        tri = Polygon([
            (mid_x, mid_y + ARROW_HEIGHT/2), 
            (mid_x - ARROW_BASE/2, mid_y - ARROW_HEIGHT/2),
            (mid_x + ARROW_BASE/2, mid_y - ARROW_HEIGHT/2)
        ])
        orientation_markers.append(tri)

    # Integrate
    GRID_STRUT_WIDTH = 6.0 
    strut_union = unary_union(final_struts).buffer(GRID_STRUT_WIDTH/2.0, cap_style=1)
    marker_union = unary_union(orientation_markers) if orientation_markers else Polygon()
    collar_union = unary_union(collar_shapes)
    
    raw_jig = unary_union([strut_union, collar_union, marker_union])
    
    # 5. Difference
    clearance_mask = parts_union.buffer(CLEARANCE)
    safe_mask = unary_union([clearance_mask, holes_mask])
    
    final_jig = raw_jig.difference(safe_mask)
    
    # Clip
    final_jig = final_jig.intersection(box(0,0, box_w, box_h))
    
    # Cleanup
    # User Report: "very small piece... too small".
    # ORDER MATTERS: Clean first (which might break thin necks), THEN remove small islands.
    final_jig, _ = clean_geometry(final_jig, min_width=2.0)
    final_jig = remove_small_islands(final_jig, min_area=150.0)
    
    return {'poly': final_jig, 'labels': []}