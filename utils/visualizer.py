# visualizer.py

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from shapely.ops import triangulate
from shapely.geometry import Polygon
from plotly.colors import sample_colorscale, get_colorscale

def make_north_arrow_mesh(box_w, box_h, box_d):
    """Creates a compact 3D mesh for a North arrow in the top-right corner."""
    ax, ay, az = box_w - 10, box_h - 10, box_d + 5
    arrow_height = 15
    shaft_radius = 0.5
    head_radius = 2
    head_height = 5
    vertices = [
        (ax, ay - shaft_radius, az), (ax, ay + shaft_radius, az),
        (ax, ay + shaft_radius, az + arrow_height - head_height), (ax, ay - shaft_radius, az + arrow_height - head_height),
        (ax, ay - head_radius, az + arrow_height - head_height), (ax, ay + head_radius, az + arrow_height - head_height),
        (ax, ay, az + arrow_height)
    ]
    x, y, z = zip(*vertices)
    i, j, k = zip(*[(0, 1, 2), (0, 2, 3), (4, 5, 6)])
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='red', opacity=1.0, name='North Arrow')

@st.cache_data
def _get_cached_layer_mesh(_polygons, z_bottom, z_top, name, geom_hash):
    if not _polygons: return None
    all_verts, all_tris = [], []
    for poly_dict in _polygons:
        # VISUAL OPTIMIZATION: Simplify significantly for 3D Viewer (0.2mm)
        # This reduces triangle count by 10x while looking identical on screen.
        poly = poly_dict['poly'].simplify(0.2, preserve_topology=True)
        if poly.is_empty: continue
        polys_to_triangulate = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        for p in polys_to_triangulate:
            top_tris = [tri for tri in triangulate(p) if tri.within(p)]
            vert_offset = len(all_verts)
            top_verts, vert_map = [], {}
            for tri in top_tris:
                for x, y in tri.exterior.coords:
                    if (x, y) not in vert_map:
                        vert_map[(x, y)] = len(top_verts)
                        top_verts.append((x, y))
            if not top_verts: continue
            all_verts.extend([(v[0], v[1], z_top) for v in top_verts])
            for tri in top_tris:
                all_tris.append([vert_offset + vert_map[c] for c in tri.exterior.coords[:3]])
            def create_walls(coords):
                wall_vert_offset = len(all_verts)
                wall_verts = []
                for x, y in coords: wall_verts.extend([(x, y, z_bottom), (x, y, z_top)])
                all_verts.extend(wall_verts)
                for i in range(len(coords) - 1):
                    v_idx = wall_vert_offset + i * 2
                    all_tris.extend([[v_idx, v_idx + 1, v_idx + 3], [v_idx, v_idx + 3, v_idx + 2]])
            create_walls(list(p.exterior.coords))
            for interior in p.interiors: create_walls(list(interior.coords))
    if not all_verts: return None
    vx, vy, vz = zip(*all_verts)
    i, j, k = zip(*all_tris)
    return go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k, opacity=1.0, flatshading=True, name=name)

def create_3d_scene(all_layer_geoms, settings, current_layer_idx, n_total_layers, show_slicer, show_wireframe, camera_override, show_axes=False, original_geoms=None, color_mode='topo'):
    box_w, box_h, box_d, mat_th = settings['box_w'], settings['box_h'], settings['box_d'], settings['mat_th']
    fig = go.Figure()

    if color_mode == 'wood':
        wood_face = '#E3C099' # Light wood
        wood_edge = '#8B4513' # Darker brown for laser cut edge
        layer_colors = [wood_face] * n_total_layers
    else:
        colorscale = get_colorscale('Earth')
        layer_colors = sample_colorscale(colorscale, np.linspace(0, 1, n_total_layers))

    if original_geoms:
        for i in range(1, current_layer_idx + 1):
            z = i * mat_th
            line_color = wood_edge if color_mode == 'wood' else layer_colors[i-1]
            for poly_dict in original_geoms[i-1]:
                poly = poly_dict['poly']
                if poly.is_empty: continue
                
                geoms_to_draw = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)

                for p in geoms_to_draw:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter3d(x=list(x), y=list(y), z=[z]*len(x), mode='lines', line=dict(color=line_color, width=3), name=f'Original Layer {i}'))
                    for interior in p.interiors:
                        x, y = interior.xy
                        fig.add_trace(go.Scatter3d(x=list(x), y=list(y), z=[z]*len(x), mode='lines', line=dict(color=line_color, width=3), showlegend=False))
    
    for i in range(1, current_layer_idx + 1):
        z_bottom, z_top = (i - 1) * mat_th, i * mat_th
        layer_polys = all_layer_geoms[i - 1]
        geom_hash = sum(p['area'] for p in layer_polys)
        layer_mesh = _get_cached_layer_mesh(tuple(layer_polys), z_bottom, z_top, f'Layer {i}', geom_hash)
        if layer_mesh:
            is_current_layer = (i == current_layer_idx)
            layer_mesh.color = 'rgba(255, 0, 0, 0.7)' if (show_slicer and is_current_layer) else layer_colors[i-1]
            if show_wireframe:
                layer_mesh.update(alphahull=0, lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.4), lightposition=dict(x=1000, y=1000, z=5000))
            elif color_mode == 'wood':
                # Use flat lighting for wood mode to prevent color shifting on rotation
                layer_mesh.update(lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0))
            fig.add_trace(layer_mesh)
    # Dowel Rendering Archived
    # if settings.get('use_dowels'):
    #     for d in settings.get('dowel_data', []):
    #         max_dowel_layer = min(n_total_layers - d.get('skip', 1), current_layer_idx)
    #         d_height = max_dowel_layer * mat_th
    #         if d_height > 0:
    #             fig.add_trace(make_cylinder_mesh(d.get('x'), d.get('y'), 0, d_height, settings.get('dowel_diam', 3.0)/2.0, color='cyan'))
    
    scene_dict = dict(
        aspectmode='data',
        xaxis=dict(visible=show_axes, title='X (mm)'),
        yaxis=dict(visible=show_axes, title='Y (mm)'),
        zaxis=dict(visible=show_axes, title='Z (mm)', range=[0, box_d + 30]),
        # Backed out zoom (2.0) to prevent edge clipping as per user request
        camera=camera_override or dict(eye=dict(x=2.0, y=-2.0, z=2.0))
    )
    fig.update_layout(showlegend=False, scene=scene_dict, margin=dict(l=0, r=0, b=0, t=0))
    return fig

def make_cylinder_mesh(x, y, z_min, z_max, radius, color='cyan', name='Dowel'):
    resolution = 16
    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    x_circ, y_circ = x + radius * np.cos(theta), y + radius * np.sin(theta)
    xs = np.concatenate([x_circ, x_circ, [x], [x]])
    ys = np.concatenate([y_circ, y_circ, [y], [y]])
    zs = np.concatenate([np.full(resolution, z_min), np.full(resolution, z_max), [z_min], [z_max]])
    i, j, k = [], [], []
    n, b_cen, t_cen = resolution, 2 * resolution, 2 * resolution + 1
    for idx in range(n):
        next_idx = (idx + 1) % n
        i.extend([[idx, next_idx, idx + n], [next_idx, next_idx + n, idx + n], [b_cen, idx, next_idx], [t_cen, idx + n, next_idx + n]])
    face_indices = np.array(i).flatten()
    return go.Mesh3d(x=xs, y=ys, z=zs, i=face_indices[::3], j=face_indices[1::3], k=face_indices[2::3], color=color, name=name)

def create_2d_view(polygons, width_mm, height_mm, polygons_to_highlight=None, previous_layer_polys=None, original_polys=None, hover_polys_with_info=None, manual_points=None, layer_color=None, problem_polys=None, active_problem_poly=None, layer_index=None, highlight_problem=None, history_highlight_polys=None, modifications_polys=None, raw_diff_geometry=None, title="2D View", show_legend=True):
    
    fig = go.Figure()
    
    # Pre-calculate bounds if not provided?
    # Actually we just use what is passed.

    # --- Render Modifications (Auto-Heal) in GREEN (Background Layer) ---
    if modifications_polys:
        # We need to define add_poly early or move this block down.
        # Let's move this block AFTER add_poly definition.
        pass

    # ... (existing setup) ...

    # Helper to add a polygon (with holes) to the figure
    def add_poly(p, line_style, fill_color, hover_text=None, show_legend=False, uid=None):
        if p.is_empty: return
        
        # Exterior
        x, y = p.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y), mode='lines',
            line=line_style, fill='toself', fillcolor=fill_color,
            hoverinfo='text' if hover_text else 'none',
            hovertext=hover_text,
            name=hover_text or '',
            showlegend=show_legend,
            uid=uid
        ))
        
        # Interiors (draw over with white to simulate holes)
        # Note: Plotly doesn't handle holes natively in 'toself' well without complex multipolygon handling (M M Z).
        # Drawing white shapes on top is a hack but works for 2D view on white background.
        for interior in p.interiors:
            xi, yi = interior.xy
            fig.add_trace(go.Scatter(
                x=list(xi), y=list(yi), mode='lines',
                line=dict(color=line_style.get('color', 'black'), width=1),
                fill='toself', fillcolor='white', # Hack for holes
                hoverinfo='skip', showlegend=False
            ))

    # --- Render Modifications (Auto-Heal) in GREEN ---
    if modifications_polys:
        for m in modifications_polys:
            # m is a dict: {'poly': poly, 'type': 'modification', 'area': area, 'uid': uid}
            g = m['poly']
            if not g.is_empty:
                # Green highlight for added mass
                uid = m.get('uid')
                # Use a specific green style
                add_poly(g, dict(color='#00CC00', width=1), 'rgba(0, 204, 0, 0.4)', hover_text=f"Auto-Heal (+{m['area']:.1f}mm²)", uid=uid)
    """Creates an interactive 2D Plotly figure for island management."""
    # fig = go.Figure() # Removed redundant init
    
    # Add manual bridge points
    if manual_points:
        xs = [p['x'] for p in manual_points]
        ys = [p['y'] for p in manual_points]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(color='purple', size=8, symbol='x'), name='Manual Bridges'))

    # Render History Highlights (Focus) - Draw LAST to be on top
    if history_highlight_polys:
         for poly in history_highlight_polys:
             if poly.is_empty: continue
             p_geoms = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
             for g in p_geoms:
                 # Exterior only for highlight
                 x, y = g.exterior.xy
                 fig.add_trace(go.Scatter(
                     x=list(x), y=list(y), mode='lines',
                     line=dict(color='#FF00FF', width=3, dash='dot'), # Magenta
                     fill='toself', fillcolor='rgba(255, 0, 255, 0.2)',
                     name='History Focus', showlegend=False, hoverinfo='skip'
                 ))

    fig.update_layout(
        xaxis=dict(range=[0, width_mm], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[0, height_mm], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    target_problem_poly = None
    if highlight_problem and problem_polys:
         # Find the problem that matches the key? 
         # Wait, problem_polys is a list of dicts. We don't have the keys here unless we passed them.
         # 'highlight_problem' passed from view is a key string.
         # Ideally we should just pass the geometry directly as 'active_problem_poly', 
         # which the view already does implicitly via logic (or should).
         pass
         
    # Actually, the View already calculates 'problem_map' and can pass the exact geometry.
    # But 'plot_layers_view' calls visualizer.plot_layers_2d which calls create_2d_view.
    # We need to thread the 'highlight_problem' KEY through to here, OR change how plot_layers_view calls this.
    
    # In 'views.py', plot_layers_view is calling:
    # visualizer.plot_layers_2d(..., highlight_problem=highlight_key)
    
    # So we need to update 'plot_layers_2d' first in visualizer.py to handle this.
    # Let's pivot: I am editing create_2d_view, but I should check plot_layers_2d first.
    pass

    # Helper to add a polygon (with holes) to the figure
    def add_poly(p, line_style, fill_color, hover_text=None, show_legend=False, uid=None):
        if p.is_empty: return
        
        # Exterior
        x, y = p.exterior.xy
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y), mode='lines',
            line=line_style, fill='toself', fillcolor=fill_color,
            hoverinfo='text' if hover_text else 'none',
            hovertext=hover_text,
            name=hover_text or '',
            showlegend=show_legend,
            uid=uid
        ))
        
        # Interiors (draw over with white to simulate holes)
        for interior in p.interiors:
            x_i, y_i = interior.xy
            fig.add_trace(go.Scatter(
                x=list(x_i), y=list(y_i), mode='lines',
                line=line_style, fill='toself', fillcolor='white',
                showlegend=False, hoverinfo='none'
            ))

    # Draw underlays first, so they appear behind
    # --- Layer Below ---
    if previous_layer_polys:
        # Pre-merge for cleaner visualization?
        try:
            from shapely.ops import unary_union
            merged_prev = unary_union([p_dict['poly'] for p_dict in previous_layer_polys])
            if not merged_prev.is_empty:
                geoms = list(merged_prev.geoms) if merged_prev.geom_type in ['MultiPolygon', 'GeometryCollection'] else [merged_prev]
                first = True
                for g in geoms:
                    if g.area < 1.0: continue
                    add_poly(g, dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dot'), None, hover_text="Layer Below", show_legend=(first and show_legend))
                    first = False
        except Exception: # Catch any shapely errors during union
            pass



    # Add original islands if requested.
    if original_polys:
        for p_dict in [p for p in original_polys if p['type'] == 'island']:
            geoms = list(p_dict['poly'].geoms) if p_dict['poly'].geom_type == 'MultiPolygon' else [p_dict['poly']]
            for g in geoms:
                add_poly(g, dict(color='rgba(0, 0, 255, 0.6)', width=1.5, dash='dot'), 'rgba(0,0,0,0)')

    # Add main layer polygons
    highlight_wkts = {p['poly'].wkt for p in (polygons_to_highlight or [])}
    for p_dict in polygons:
        geoms = list(p_dict['poly'].geoms) if p_dict['poly'].geom_type == 'MultiPolygon' else [p_dict['poly']]
        for g in geoms:
            default_fill = layer_color if layer_color else 'rgba(139, 69, 19, 0.7)' # Fallback to SaddleBrown
            
            # Ensure the fill color has transparency so underlays are visible.
            # The layer_color from the colorscale is opaque by default (e.g., 'rgb(r,g,b)').
            if default_fill and default_fill.startswith('rgb('):
                default_fill = default_fill.replace('rgb', 'rgba').replace(')', ', 0.7)')

            is_highlighted = g.wkt in highlight_wkts
            line = dict(color='blue' if is_highlighted else '#663300', width=2 if is_highlighted else 1)
            fill = 'rgba(100, 149, 237, 0.5)' if is_highlighted else default_fill # CornflowerBlue for highlight
            # Generate pseudo-stable ID
            cx, cy = int(g.centroid.x * 10), int(g.centroid.y * 10)
            p_dict['uid'] = f"poly_{cx}_{cy}_{int(g.area)}"
            add_poly(g, line, fill, uid=p_dict['uid'])

    # Add problem polygons (active vs inactive)
    if problem_polys:
        # If we have a highlight_problem KEY, we need to find the matching poly.
        # But wait, problem_polys here is just a list of dicts. We don't have keys.
        # We rely on 'active_problem_poly' passed in.
        
        # If 'active_problem_poly' is passed, we highlight it.
        # The calling function needs to resolve the key to a poly.
        
        for p_dict in problem_polys:
            is_active = False
            if active_problem_poly:
                is_active = p_dict['poly'].equals(active_problem_poly)
            
            geoms = list(p_dict['poly'].geoms) if p_dict['poly'].geom_type == 'MultiPolygon' else [p_dict['poly']]
            for g in geoms:
                color = 'red' if is_active else 'orange'
                width = 3 if is_active else 1
                fill = 'rgba(255, 0, 0, 0.8)' if is_active else 'rgba(255, 165, 0, 0.5)'
                # Use orig_idx if available for problems
                uid = p_dict.get('orig_idx')
                if not uid:
                    cx, cy = int(g.centroid.x * 10), int(g.centroid.y * 10)
                    uid = f"prob_{cx}_{cy}"
                p_dict['uid'] = uid
                add_poly(g, dict(color=color, width=width, dash='solid'), fill, hover_text="Thin Feature", uid=uid)

    # --- RAW DIFF OVERLAY (Moved to End for Visibility) ---
    if raw_diff_geometry and not raw_diff_geometry.is_empty:
        # Provide a Red Dashed Overlay of the original geometry to show changes
        # We use a pure line (fill=None) to ensure we see the geometry below too.
        geoms = list(raw_diff_geometry.geoms) if raw_diff_geometry.geom_type in ['MultiPolygon', 'GeometryCollection'] else [raw_diff_geometry]
        first = True
        for g in geoms:
             # Pass None as fill_color to ensure transparency
             # MAGENTA SOLID: Impossible to miss.
             add_poly(g, dict(color='#FF00FF', width=3, dash='solid'), None, hover_text="Original Raw Geometry", show_legend=first)
             first = False

    # Add hover-able, transparent shapes for island info
    if hover_polys_with_info:
        for p_info in hover_polys_with_info:
            geoms = list(p_info['poly'].geoms) if p_info['poly'].geom_type == 'MultiPolygon' else [p_info['poly']]
            for g in geoms:
                add_poly(g, dict(width=0), 'rgba(0,0,0,0)', hover_text=p_info['info'])

    # Add manual bridge points
    if manual_points:
        xs = [p['x'] for p in manual_points]
        ys = [p['y'] for p in manual_points]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(color='magenta', size=10, symbol='cross'), name='Manual Bridges', hoverinfo='none'))

    # Add a smaller, stylized North Arrow in the top-right corner
    arrow_x = width_mm - 15      # Position from right edge
    arrow_y_base = height_mm - 20  # Position from top edge
    # Black part of the arrow (points North)
    fig.add_shape(type="path",
        path=f"M {arrow_x},{arrow_y_base+9} L {arrow_x-3},{arrow_y_base-3} L {arrow_x},{arrow_y_base} L {arrow_x+3},{arrow_y_base-3} Z",
        fillcolor="black", line_color="black"
    )
    # Gray part of the arrow (points South)
    fig.add_shape(type="path",
        path=f"M {arrow_x},{arrow_y_base-9} L {arrow_x-3},{arrow_y_base+3} L {arrow_x},{arrow_y_base} L {arrow_x+3},{arrow_y_base+3} Z",
        fillcolor="#aaaaaa", line_color="black"
    )
    fig.add_annotation(
        x=arrow_x, y=arrow_y_base + 12,
        text="N", showarrow=False,
        font=dict(size=12, color="black", family="serif")
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[0, width_mm], showgrid=True, gridcolor='#eee', zeroline=False, title='X (mm)'),
        yaxis=dict(range=[0, height_mm], showgrid=True, gridcolor='#eee', zeroline=False, scaleanchor="x", scaleratio=1, title='Y (mm)'),
        margin=dict(l=0, r=0, b=20, t=40),
        plot_bgcolor='white',
        height=700,
        title=dict(text=f"2D View - Layer {layer_index}" if layer_index else "2D Preview", y=0.98),
        hovermode='closest',
        uirevision='topo_persistent_view', # Preserves zoom/pan state consistently across all updates and layers
    )
    return fig

def plot_layers_2d(all_layer_geoms, box_w, box_h, show_layer=None, highlight_problem=None):
    """
    Wrapper for create_2d_view that handles resolving layer data.
    highlight_problem: str (Key of problem to highlight, e.g. 'L5_P2')
    """
    if show_layer is None: return go.Figure()
    
    layer_idx = show_layer - 1 
    if layer_idx < 0 or layer_idx >= len(all_layer_geoms): return go.Figure()
    
    current_layer = all_layer_geoms[layer_idx]
    
    # Separation
    polys = [p for p in current_layer if p['type'] != 'problem']
    problems = [p for p in current_layer if p['type'] == 'problem']
    
    # Resolve Highlight
    active_problem_poly = None
    if highlight_problem:
        # Check if the highlight key belongs to this layer
        # Key format: L{layer}_...
        prefix = f"L{show_layer}_"
        if highlight_problem.startswith(prefix):
            # Find the match
            for j, p in enumerate(problems):
                key = f"L{show_layer}_M{p['manual_idx']}" if p.get('is_manual') else f"L{show_layer}_P{j}"
                if key == highlight_problem:
                    active_problem_poly = p['poly']
                    break

    # Previous layer for context
    prev_layer = all_layer_geoms[layer_idx - 1] if layer_idx > 0 else None
    
    return create_2d_view(
        polys, box_w, box_h, 
        previous_layer_polys=prev_layer,
        problem_polys=problems,
        active_problem_poly=active_problem_poly,
        layer_index=show_layer
    )