import io
import zipfile
import base64
import numpy as np
import math
import ezdxf
from ezdxf.render import MeshBuilder
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.backends.backend_ps import FigureCanvasPS
from matplotlib.patches import Polygon as MplPolygon, PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon, Point
from shapely import affinity
from utils import geometry_engine

# --- SHARED HELPERS ---
def create_patch(poly, **kwargs):
    vertices = []
    codes = []
    
    # Helper for a single ring
    def process_ring(x_coords, y_coords):
        if len(x_coords) < 3: return # Degenerate
        ring_codes = [Path.MOVETO] + [Path.LINETO] * (len(x_coords) - 2) + [Path.CLOSEPOLY]
        return np.column_stack((x_coords, y_coords)), ring_codes

    if poly.is_empty: return PathPatch(Path([(0,0)], [Path.MOVETO]), **kwargs)

    x, y = poly.exterior.xy
    res = process_ring(x, y)
    if not res:
        return PathPatch(Path([(0,0)], [Path.MOVETO]), **kwargs)
    
    v, c = res
    vertices.append(v)
    codes.append(c)
    
    for interior in poly.interiors:
        x, y = interior.xy
        res = process_ring(x, y)
        if res:
            v, c = res
            vertices.append(v)
            codes.append(c)
    
    if not codes: return PathPatch(Path([(0,0)], [Path.MOVETO]), **kwargs)
    
    flat_verts = np.concatenate(vertices)
    flat_codes = np.concatenate(codes).astype(np.uint8)
    
    path = Path(flat_verts, flat_codes)
    return PathPatch(path, **kwargs)

def generate_assembly_guide_pdf(final_geoms, settings, jig_modifications_dict, current_dowels):
    buffer = io.BytesIO()
    w_mm, h_mm = settings['box_w'], settings['box_h']
    
    wood_base = '#E3C099'
    jig_face = '#de2d26' 
    jig_edge = '#a50f15'
    
    def adjust_color(hex_color, factor):
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = [min(255, max(0, int(c * factor))) for c in rgb]
        return f"#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}"

    wood_edge = adjust_color(wood_base, 0.6)
    iso_cos = math.cos(math.radians(30))
    iso_sin = math.sin(math.radians(30))
    
    def get_iso_transform(z):
        return [iso_cos, -iso_cos, iso_sin, iso_sin, 0, z]

    def draw_iso_shadowbox(ax, w, h, d):
        floor_poly = Polygon([(0,0), (w,0), (w,h), (0,h)])
        floor_proj = affinity.affine_transform(floor_poly, get_iso_transform(0))
        ax.add_patch(create_patch(floor_proj, facecolor='#333333', edgecolor='none', alpha=0.1))
        ax.add_patch(create_patch(floor_proj, facecolor='none', edgecolor='#333333', linewidth=1, linestyle='--'))

    with PdfPages(buffer) as pdf:
        # --- STYLING CONSTANTS ---
        C_PRIMARY = '#2c3e50'
        C_ACCENT = '#e74c3c'
        C_WOOD = '#E3C099'
        C_JIG = '#e67e22'
        C_TEXT = '#34495e'
        C_GHOST = '#ecf0f1'
        FONT_MAIN = 'sans-serif'
        
        def draw_header(fig, title, subtitle, progress=0.0):
            ax_head = fig.add_axes([0, 0.90, 1, 0.10])
            ax_head.axis('off')
            ax_head.add_patch(MplPolygon([(0,0), (1,0), (1,1), (0,1)], transform=ax_head.transAxes, fc=C_PRIMARY, ec='none'))
            fig.text(0.05, 0.95, title, fontsize=24, color='white', weight='bold', fontfamily=FONT_MAIN, va='center')
            fig.text(0.95, 0.95, subtitle, fontsize=14, color='#bdc3c7', ha='right', fontfamily=FONT_MAIN, va='center')
            if progress > 0:
                ax_head.add_patch(MplPolygon([(0,0), (progress,0), (progress,0.05), (0,0.05)], transform=ax_head.transAxes, fc=C_ACCENT, ec='none'))

        def draw_footer(fig, page_num):
            fig.text(0.5, 0.02, f"Page {page_num} • {settings.get('proj_name', 'Project')} Assembly Guide", ha='center', fontsize=8, color='#95a5a6', fontfamily=FONT_MAIN)

        # --- COVER PAGE ---
        fig = Figure(figsize=(8.27, 11.69))
        ax_iso = fig.add_axes([0.1, 0.35, 0.8, 0.45])
        ax_iso.axis('off'); ax_iso.set_aspect('equal')
        total_h = len(final_geoms) * settings['mat_th']
        
        n_layers = len(final_geoms)
        for i, layer_polys in enumerate(final_geoms):
             z_height = i * settings['mat_th']
             factor = 0.75 + (0.35 * i / max(1, n_layers - 1))
             layer_face = adjust_color(C_WOOD, factor)
             for p in layer_polys:
                if 'poly' in p and not p['poly'].is_empty:
                    polys = [p['poly']] if p['poly'].geom_type == 'Polygon' else list(p['poly'].geoms)
                    for sub in polys:
                        iso_poly = affinity.affine_transform(sub, get_iso_transform(z_height))
                        ax_iso.add_patch(create_patch(iso_poly, facecolor=layer_face, edgecolor=adjust_color(layer_face, 0.8), linewidth=0.3))
        ax_iso.autoscale_view()
        
        draw_header(fig, "ASSEMBLY GUIDE", "TopoBox Pro", 0)
        fig.text(0.1, 0.28, settings.get('proj_name', 'My Project').upper(), fontsize=36, weight='bold', color=C_PRIMARY, fontfamily=FONT_MAIN)
        fig.text(0.1, 0.24, "3D Topographic Model", fontsize=16, color=C_TEXT, fontfamily=FONT_MAIN)
        
        ax_stats = fig.add_axes([0.1, 0.08, 0.8, 0.12])
        ax_stats.axis('off')
        stats = [("DIMENSIONS", f"{w_mm:.0f} x {h_mm:.0f} mm"), ("LAYERS", f"{n_layers}"), ("THICKNESS", f"{settings['mat_th']} mm"), ("TOTAL HEIGHT", f"{total_h:.1f} mm")]
        for idx, (label, val) in enumerate(stats):
            x_pos = idx / len(stats)
            ax_stats.text(x_pos, 0.6, label, fontsize=8, color='#95a5a6', weight='bold')
            ax_stats.text(x_pos, 0.3, val, fontsize=14, color=C_PRIMARY, weight='bold')

        fig.text(0.9, 0.02, "Generated by TopoBox", ha='right', fontsize=8, color='#bdc3c7')
        pdf.savefig(fig)
        
        # --- LAYER STEPS ---
        for i, layer in enumerate(final_geoms):
            layer_num = i + 1
            jig_mods = jig_modifications_dict.get(f"L{layer_num}", [])
            jig_data = geometry_engine.generate_jig_geometry(layer, current_dowels, w_mm, h_mm, layer_num, jig_mods, conn_width=settings['jig_conn_width'], grid_spacing=settings.get('jig_grid_spacing', 20.0), fluid_smoothing=settings.get('jig_fluid', True))
            
            fig = Figure(figsize=(8.27, 11.69))
            draw_header(fig, f"STEP {layer_num}", f"Layer {layer_num} of {n_layers}", progress=layer_num/n_layers)
            
            # Left Rail: Parts and Jig (Flat View)
            # 1. Parts
            ax_parts = fig.add_axes([0.05, 0.55, 0.40, 0.25])
            ax_parts.set_aspect('equal'); ax_parts.axis('off')
            ax_parts.set_title("1. Collect Parts", loc='left', fontsize=10, color=C_ACCENT, weight='bold')
            ax_parts.add_patch(MplPolygon([(0,0), (w_mm,0), (w_mm,h_mm), (0,h_mm)], closed=True, fc='#f9f9f9', ec='#ecf0f1', linestyle='-'))
            
            part_count = 0
            for p in layer:
                if 'poly' in p and not p['poly'].is_empty:
                    part_count += 1
                    polys = [p['poly']] if p['poly'].geom_type == 'Polygon' else list(p['poly'].geoms)
                    for sub in polys:
                        ax_parts.add_patch(create_patch(sub, facecolor=C_WOOD, edgecolor=adjust_color(C_WOOD, 0.6), linewidth=0.5))
            ax_parts.autoscale_view()
            
            # 2. Jig
            ax_jig = fig.add_axes([0.05, 0.20, 0.40, 0.25])
            ax_jig.set_aspect('equal'); ax_jig.axis('off')
            ax_jig.set_title("2. Place Alignment Jig", loc='left', fontsize=10, color=C_ACCENT, weight='bold')
            ax_jig.add_patch(MplPolygon([(0,0), (w_mm,0), (w_mm,h_mm), (0,h_mm)], closed=True, fc='#f9f9f9', ec='#ecf0f1', linestyle='-'))
            
            if jig_data:
                 jpolys = [jig_data['poly']] if jig_data['poly'].geom_type == 'Polygon' else list(jig_data['poly'].geoms)
                 for jp in jpolys:
                     ax_jig.add_patch(create_patch(jp, facecolor=C_JIG, edgecolor=adjust_color(C_JIG,0.8), alpha=0.9, linewidth=0.5))
            else:
                 ax_jig.text(w_mm/2, h_mm/2, "No Jig Needed", ha='center', color='#bdc3c7')
            ax_jig.autoscale_view()
            
            # Right Rail: Assembly View (Iso)
            ax_iso = fig.add_axes([0.50, 0.20, 0.45, 0.60])
            ax_iso.set_aspect('equal'); ax_iso.axis('off')
            ax_iso.set_title("3. Assembly", loc='left', fontsize=10, color=C_ACCENT, weight='bold')
            draw_iso_shadowbox(ax_iso, w_mm, h_mm, z_height + settings['mat_th'] + 5)
            
            for stack_i in range(i + 1):
                is_current = (stack_i == i)
                z = stack_i * settings['mat_th']
                if is_current: fc, ec, lw, alpha = C_WOOD, C_PRIMARY, 0.8, 1.0
                else: fc, ec, lw, alpha = '#f0f0f0', '#dcdcdc', 0.3, 0.6
                for p in final_geoms[stack_i]:
                    if 'poly' in p and not p['poly'].is_empty:
                        polys = [p['poly']] if p['poly'].geom_type == 'Polygon' else list(p['poly'].geoms)
                        for sub in polys:
                            iso_P = affinity.affine_transform(sub, get_iso_transform(z))
                            ax_iso.add_patch(create_patch(iso_P, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha))
            ax_iso.autoscale_view()
            
            instructions = [f"1. Gather {part_count} part(s) for Layer {layer_num}.", "2. Place the Orange Jig frame on top of the previous layer." if jig_data else "2. No Jig required for this layer. Align visually.", "3. Apply glue to the previous layer where parts will sit.", "4. Drop parts into the Jig openings.", "5. Remove Jig carefully (if reusable) or leave for clamping."]
            fig.text(0.05, 0.15, "\n".join(instructions), fontsize=10, color=C_TEXT, fontfamily='monospace', va='top')
            draw_footer(fig, i+2)
            pdf.savefig(fig)
    return buffer.getvalue()

def generate_zip_data(final_geoms, settings, jig_modifications_dict, current_dowels, export_fmt, part_color='#E3C099'):
    zip_buffer = io.BytesIO()
    w_mm, h_mm = settings['box_w'], settings['box_h']
    proj_name = settings.get('proj_name', 'Project')
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, layer in enumerate(final_geoms):
            layer_num = i + 1
            jig_mods = jig_modifications_dict.get(f"L{layer_num}", [])
            jig_data = geometry_engine.generate_jig_geometry(layer, current_dowels, w_mm, h_mm, layer_num, jig_mods, conn_width=settings['jig_conn_width'], grid_spacing=settings.get('jig_grid_spacing', 20.0), fluid_smoothing=settings.get('jig_fluid', True))
            
            if export_fmt == "SVG":
                pass 
            elif export_fmt == "DXF":
                doc = ezdxf.new(); msp = doc.modelspace()
                doc.layers.new(name='EXTERIOR', dxfattribs={'color': 4}) # Cyan
                doc.layers.new(name='INTERIOR', dxfattribs={'color': 1}) # Red
                
                # Combine parts and jig for this layer (if any)
                all_polys = []
                for p in layer:
                    geom = p['poly']
                    if geom.is_empty: continue
                    sub = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                    all_polys.extend(sub)
                
                if jig_data:
                    jgeom = jig_data['poly']
                    if not jgeom.is_empty:
                        sub = [jgeom] if jgeom.geom_type == 'Polygon' else list(jgeom.geoms)
                        all_polys.extend(sub)
                        
                for poly in all_polys:
                    if poly.is_empty: continue
                    msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': 'EXTERIOR'})
                    for interior in poly.interiors: 
                        msp.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'layer': 'INTERIOR'})
                        
                dxf_stream = io.StringIO(); doc.write(dxf_stream)
                zf.writestr(f"Components/{proj_name}_L{layer_num}_Parts.dxf", dxf_stream.getvalue())
            
            elif export_fmt in ["PDF", "PNG", "JPG", "EPS"]:
                # 1:1 Scale (mm to inches)
                fig = Figure(figsize=(w_mm / 25.4, h_mm / 25.4))
                
                # Select correct backend canvas
                if export_fmt == "PDF":
                    canvas = FigureCanvasPdf(fig)
                elif export_fmt == "EPS":
                    canvas = FigureCanvasPS(fig)
                else:
                    canvas = FigureCanvasAgg(fig)

                ax = fig.add_subplot(111)
                ax.set_xlim(0, w_mm); ax.set_ylim(0, h_mm); ax.set_aspect('equal'); ax.axis('off')

                # Parts (use create_patch for holes)
                for p in layer:
                    geom = p['poly']
                    if geom.is_empty: continue
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                    for poly in polys:
                        if not poly.is_empty:
                             ax.add_patch(create_patch(poly, facecolor=part_color, edgecolor='black', linewidth=1))
                
                # Jigs (use create_patch for holes)
                if jig_data:
                    jig_poly = jig_data['poly']
                    polys = [jig_poly] if jig_poly.geom_type == 'Polygon' else list(jig_poly.geoms)
                    for poly in polys:
                        if not poly.is_empty:
                             ax.add_patch(create_patch(poly, facecolor='#de2d26', edgecolor='#a50f15', linewidth=1, alpha=0.8))
                
                # Add Metadata Text - REMOVED per user request
                # ax.text(5, 5, f"Sheet Size: {w_mm:.0f}x{h_mm:.0f}mm | {proj_name} | Layer {layer_num}", fontsize=8, color='#333333', va='bottom', ha='left')

                img_data = io.BytesIO()
                canvas.print_figure(img_data, format=export_fmt.lower(), bbox_inches='tight', pad_inches=0.1)
                zf.writestr(f"Components/{proj_name}_L{layer_num}_Parts.{export_fmt.lower()}", img_data.getvalue())

    return zip_buffer.getvalue()

# --- 3D HELPERS ---
from scipy.spatial import Delaunay

def triangulate_polygon(poly):
    """
    Triangulates a shapely Polygon (handling holes) using constrained Delaunay-like concept.
    Returns: (vertices, triangles)
    vertices: Nx3 numpy array (z=0)
    triangles: Mx3 integer array of indices
    """
    if poly.is_empty: return np.empty((0,3)), np.empty((0,3), dtype=int)
    
    # 1. Flatten all vertices (exterior + interiors)
    boundary_coords = list(poly.exterior.coords)[:-1] # Drop duplicate end
    hole_start_indices = []
    current_idx = len(boundary_coords)
    
    all_coords = list(boundary_coords)
    
    for interior in poly.interiors:
        hole_coords = list(interior.coords)[:-1]
        if not hole_coords: continue
        all_coords.extend(hole_coords)
        hole_start_indices.append(current_idx)
        current_idx += len(hole_coords)
    
    points_2d = np.array(all_coords)
    
    # 2. Triangulate all points
    if len(points_2d) < 3: return np.empty((0,3)), np.empty((0,3), dtype=int)
    
    tri = Delaunay(points_2d)
    
    # 3. Filter triangles that are OUTSIDE the polygon (holes or concave hull)
    # We check the centroid of each triangle
    tri_centers = np.mean(points_2d[tri.simplices], axis=1)
    
    # Use Shapely's contains for robustness
    # Optimization: Vectorized containment checks are hard with Shapely.
    # Simple loop is fine for typical part complexity.
    valid_mask = []
    for c in tri_centers:
        p = Point(c[0], c[1])
        valid_mask.append(poly.contains(p))
        
    valid_simplices = tri.simplices[valid_mask]
    
    # Return 3D vertices (z=0)
    vertices_3d = np.column_stack((points_2d, np.zeros(len(points_2d))))
    
    return vertices_3d, valid_simplices

def generate_extruded_mesh_data(polys_list, thickness_mm):
    """
    Generates a single mesh for a list of polygons.
    Returns: (vertices, faces) for OBJ export.
    faces are 1-based indices if typical OBJ, but we return 0-based here for logic.
    """
    all_verts = []
    all_faces = []
    v_offset = 0
    
    for poly in polys_list:
        if poly.is_empty: continue
        sub_polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        
        for p in sub_polys:
            # 1. Triangulate Top Face (z=thickness)
            verts, tris = triangulate_polygon(p)
            if len(verts) == 0: continue
            
            n_v = len(verts)
            
            # Top vertices (z = thickness)
            top_verts = verts.copy()
            top_verts[:, 2] = thickness_mm
            
            # Bottom vertices (z = 0)
            bot_verts = verts.copy()
            bot_verts[:, 2] = 0
            
            # Current chunk vertices: [Top... , Bot...]
            chunk_verts = np.vstack((top_verts, bot_verts))
            
            # Faces: Top (CCW), Bottom (CW)
            # Top: direct indices
            top_faces = tris.copy() + v_offset
            
            # Bottom: offset by n_v, flip winding order [0, 2, 1]
            bot_faces = (tris.copy() + v_offset + n_v)[:, [0, 2, 1]]
            
            # Side Walls (Quads -> 2 Tris)
            # Must follow the rings (exterior + interiors)
            rings = [p.exterior] + list(p.interiors)
            side_faces = []
            
            # Map spatial coordinates back to indices to "sew" the sides
            # This is slightly expensive; robust way is to track ring implementation.
            # Simplified: Find vertex index for each ring coord using distance
            # KDTree is overkill, just brute force for < 1000 points.
            
            def get_idx(pt):
                # Returns index in 'verts' (0 to n_v-1) matching pt x,y
                dists = np.sum((verts[:, :2] - [pt[0], pt[1]])**2, axis=1)
                return np.argmin(dists)
            
            for ring in rings:
                coords = list(ring.coords)
                if len(coords) < 3: continue
                # Remove duplicate end if present
                if coords[0] == coords[-1]: coords.pop()
                
                n_ring = len(coords)
                for i in range(n_ring):
                    curr_pt = coords[i]
                    next_pt = coords[(i+1)%n_ring]
                    
                    idx_curr = get_idx(curr_pt)
                    idx_next = get_idx(next_pt)
                    
                    # Indices in the chunk
                    # Top: idx
                    # Bot: idx + n_v
                    
                    # Quad: TopCurr, TopNext, BotNext, BotCurr
                    # Tris: (TopCurr, BotNext, BotCurr), (TopCurr, TopNext, BotNext) - CCW outward?
                    # Let's visualize: 
                    # If we walk CCW on exterior, Normal points out.
                    # Wall: Curr -> Next. 
                    # Tr1: TopCurr, BotCurr, BotNext
                    # Tr2: TopCurr, BotNext, TopNext
                    
                    tc = idx_curr + v_offset
                    tn = idx_next + v_offset
                    bc = idx_curr + v_offset + n_v
                    bn = idx_next + v_offset + n_v
                    
                    side_faces.append([tc, bc, bn])
                    side_faces.append([tc, bn, tn])
            
            all_verts.append(chunk_verts)
            all_faces.extend(top_faces)
            all_faces.extend(bot_faces)
            all_faces.extend(side_faces)
            
            v_offset += len(chunk_verts)

    if not all_verts: return np.empty((0,3)), np.empty((0,3), dtype=int)
    return np.vstack(all_verts), np.array(all_faces)

def generate_nested_zip(nested_components, nested_jigs, nested_comp_dims, nested_jig_dims, export_fmt, ex_wood_color, ex_edge_color, proj_name="Project", mat_th=3.0):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        def write_sheets(sheets, prefix, sheet_dims):
            if not sheet_dims or sheet_dims[0] == 0 or sheet_dims[1] == 0: return
            ns_w, ns_h = sheet_dims
            for i, sheet in enumerate(sheets):
                sheet_polys = []
                dxf_parts = []
                dxf_jigs = []
                
                for placed in sheet:
                    poly = placed['data']['poly']
                    item_type = placed['data'].get('type', 'part')
                    rot = placed.get('rotation', 0)
                    if rot != 0: poly = affinity.rotate(poly, rot, origin='center')
                    minx, miny, _, _ = poly.bounds
                    dx = placed['x'] - minx; dy = placed['y'] - miny
                    moved_poly = affinity.translate(poly, xoff=dx, yoff=dy)
                    entry = {'poly': moved_poly, 'type': item_type}
                    sheet_polys.append(entry)
                    if item_type == 'jig': dxf_jigs.append(moved_poly)
                    else: dxf_parts.append(moved_poly)
                
                # Filename with Project Name
                base_filename = f"{prefix}/{proj_name}_Sheet_{i+1}"

                if export_fmt == "SVG":
                    fill_col = {'part': ex_wood_color, 'jig': '#de2d26', 'default': ex_wood_color}
                    svg = geometry_engine.generate_svg_string(sheet_polys, ns_w, ns_h, fill_color=fill_col, stroke_color=ex_edge_color, add_background=False)
                    zf.writestr(f"{base_filename}.svg", svg)

                elif export_fmt == "DXF":
                    doc = ezdxf.new(); msp = doc.modelspace()
                    doc.layers.new(name='EXTERIOR', dxfattribs={'color': 4}) # Cyan
                    doc.layers.new(name='INTERIOR', dxfattribs={'color': 1}) # Red
                    
                    all_polys_sheet = []
                    # Process Parts
                    for p in dxf_parts:
                        sub = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        all_polys_sheet.extend(sub)
                    # Process Jigs
                    for p in dxf_jigs:
                        sub = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        all_polys_sheet.extend(sub)
                        
                    for poly in all_polys_sheet:
                        if poly.is_empty: continue
                        msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': 'EXTERIOR'})
                        for interior in poly.interiors: 
                            msp.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'layer': 'INTERIOR'})
                    
                    # 1. Write Clean DXF (Cutting Only)
                    dxf_stream = io.StringIO(); doc.write(dxf_stream)
                    zf.writestr(f"{base_filename}.dxf", dxf_stream.getvalue())

                    # 2. Add Labels for Reference DXF
                    doc.layers.new(name='LABELS', dxfattribs={'color': 7}) # White/Black
                    for placed in sheet:
                        poly_geom = placed['data']['poly']
                        original_poly = placed['data']['poly']
                        rot = placed.get('rotation', 0)
                        
                        # Get oriented centroid
                        p_rot = affinity.rotate(original_poly, rot, origin='center') if rot != 0 else original_poly
                        minx, miny, _, _ = p_rot.bounds
                        dx = placed['x'] - minx
                        dy = placed['y'] - miny
                        
                        cx, cy = p_rot.centroid.x + dx, p_rot.centroid.y + dy
                        
                        # Generate Label Text
                        l_num = placed['data'].get('layer', '?')
                        if placed['data'].get('type') == 'jig':
                            label = f"L{l_num}-JIG"
                        else:
                            p_id = placed['data'].get('id', '?')
                            label = f"L{l_num}-P{p_id}"
                        
                        from ezdxf.enums import TextEntityAlignment
                        msp.add_text(label, dxfattribs={'layer': 'LABELS', 'height': 5.0}).set_placement((cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)
                            
                    # 3. Write Reference DXF (With Labels)
                    dxf_stream_ref = io.StringIO(); doc.write(dxf_stream_ref)
                    zf.writestr(f"{base_filename}_Ref.dxf", dxf_stream_ref.getvalue())

                elif export_fmt == "DXF_3D":
                    # Generate 3D Mesh DXF (Solid/Filled appearance)
                    doc = ezdxf.new()
                    msp = doc.modelspace()
                    doc.layers.new(name='PARTS_3D', dxfattribs={'color': 1})
                    
                    all_geoms = dxf_parts + dxf_jigs
                    verts, faces = generate_extruded_mesh_data(all_geoms, mat_th)
                    
                    if len(verts) > 0 and len(faces) > 0:
                        # Create Polyface Mesh using MeshBuilder (robust)
                        mesh_builder = MeshBuilder()
                        mesh_builder.add_mesh(vertices=verts.tolist(), faces=faces.tolist())
                        mesh_builder.render_polyface(msp, dxfattribs={'layer': 'PARTS_3D', 'color': 1})
                    
                    dxf_stream = io.StringIO()
                    doc.write(dxf_stream)
                    zf.writestr(f"{base_filename}_3D.dxf", dxf_stream.getvalue())

                elif export_fmt in ["PDF", "PNG", "JPG", "EPS"]:
                    # 1:1 Scale (mm to inches)
                    fig = Figure(figsize=(ns_w / 25.4, ns_h / 25.4))
                    
                    # Select correct backend canvas for vector or raster output
                    if export_fmt == "PDF":
                        canvas = FigureCanvasPdf(fig)
                    elif export_fmt == "EPS":
                        canvas = FigureCanvasPS(fig)
                    else: # PNG, JPG
                        canvas = FigureCanvasAgg(fig)

                    ax = fig.add_subplot(111)
                    ax.set_xlim(0, ns_w); ax.set_ylim(0, ns_h); ax.set_aspect('equal'); ax.axis('off')
                    ax.add_patch(MplPolygon([(0,0), (ns_w,0), (ns_w,ns_h), (0,ns_h)], closed=True, fc='none', ec='#cccccc', linestyle='--'))
                    
                    for p in dxf_parts:
                        polys = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        for poly in polys:
                            if not poly.is_empty:
                                ax.add_patch(create_patch(poly, facecolor=ex_wood_color, edgecolor='black', linewidth=1))

                    for p in dxf_jigs:
                        polys = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        for poly in polys:
                            if not poly.is_empty:
                                 ax.add_patch(create_patch(poly, facecolor='#de2d26', edgecolor='#a50f15', linewidth=1, alpha=0.8))
                    
                    # Add Metadata Text - REMOVED per user request
                    # ax.text(5, 5, f"Sheet Size: {ns_w:.0f}x{ns_h:.0f}mm | {proj_name} | Sheet {i+1}", fontsize=8, color='#333333', va='bottom', ha='left')

                    img_data = io.BytesIO()
                    canvas.print_figure(img_data, format=export_fmt.lower(), bbox_inches='tight', pad_inches=0.1)
                    zf.writestr(f"{base_filename}.{export_fmt.lower()}", img_data.getvalue())
        
        if nested_components and nested_comp_dims: write_sheets(nested_components, "Components/Nested", nested_comp_dims)
        if nested_jigs and nested_jig_dims: write_sheets(nested_jigs, "Jigs/Nested", nested_jig_dims)
    return buf.getvalue()