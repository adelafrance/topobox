import io
import zipfile
import base64
import numpy as np
import math
import ezdxf
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon as MplPolygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon, Point
from shapely import affinity
from utils import geometry_engine

def generate_assembly_guide_pdf(final_geoms, settings, jig_modifications_dict, current_dowels):
    buffer = io.BytesIO()
    w_mm, h_mm = settings['box_w'], settings['box_h']
    wood_base = '#E3C099'
    jig_face = '#de2d26' 
    jig_edge = '#a50f15'

    def create_patch(poly, **kwargs):
        vertices = []
        codes = []
        
        # Helper for a single ring
        def process_ring(x_coords, y_coords):
            if len(x_coords) < 3: return # Degenerate
            # Explicitly construct codes
            # Path expects len(codes) match len(vertices)
            # MOVETO, LINETO... LINETO, CLOSEPOLY
            # The last vertex in Shapely is usually same as first.
            # Matplotlib CLOSEPOLY ignores the vertex but needs a placeholder.
            ring_codes = [Path.MOVETO] + [Path.LINETO] * (len(x_coords) - 2) + [Path.CLOSEPOLY]
            return np.column_stack((x_coords, y_coords)), ring_codes

        x, y = poly.exterior.xy
        res = process_ring(x, y)
        if not res:
            return PathPatch(Path([(0,0)], [Path.MOVETO]), **kwargs) # Return dummy if exterior is degenerate
        
        v, c = res
        vertices.append(v)
        codes.append(c)
        
        for interior in poly.interiors:
            x, y = interior.xy
            # Check length inside helper
            res = process_ring(x, y)
            if res:
                v, c = res
                vertices.append(v)
                codes.append(c)
        
        if not codes: return PathPatch(Path([(0,0)], [Path.MOVETO]), **kwargs) # Return dummy
        
        # Flatten and ensure type
        flat_verts = np.concatenate(vertices)
        flat_codes = np.concatenate(codes).astype(np.uint8) # Explicit typing
        
        path = Path(flat_verts, flat_codes)
        return PathPatch(path, **kwargs)

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
        # Reverted to simple floor shadow per request
        floor_poly = Polygon([(0,0), (w,0), (w,h), (0,h)])
        floor_proj = affinity.affine_transform(floor_poly, get_iso_transform(0))
        ax.add_patch(create_patch(floor_proj, facecolor='#333333', edgecolor='none', alpha=0.1))
        ax.add_patch(create_patch(floor_proj, facecolor='none', edgecolor='#333333', linewidth=1, linestyle='--'))

    with PdfPages(buffer) as pdf:
        # ... (Styling constants) ...
        # --- STYLING CONSTANTS ---
        C_PRIMARY = '#2c3e50'   # Slate Blue (Headers)
        C_ACCENT = '#e74c3c'    # Red (Highlights)
        C_WOOD = '#E3C099'      # Wood Base
        C_JIG = '#e67e22'       # Orange (Jigs)
        C_TEXT = '#34495e'
        C_GHOST = '#ecf0f1'     # Previous layers (Ghosted)
        FONT_MAIN = 'sans-serif'
        
        # --- HELPERS ---
        def draw_header(fig, title, subtitle, progress=0.0):
            # Banner Background
            ax_head = fig.add_axes([0, 0.90, 1, 0.10])
            ax_head.axis('off')
            ax_head.add_patch(MplPolygon([(0,0), (1,0), (1,1), (0,1)], transform=ax_head.transAxes, fc=C_PRIMARY, ec='none'))
            
            # Text
            fig.text(0.05, 0.95, title, fontsize=24, color='white', weight='bold', fontfamily=FONT_MAIN, va='center')
            fig.text(0.95, 0.95, subtitle, fontsize=14, color='#bdc3c7', ha='right', fontfamily=FONT_MAIN, va='center')
            
            # Progress Bar (at bottom of banner)
            if progress > 0:
                ax_head.add_patch(MplPolygon([(0,0), (progress,0), (progress,0.05), (0,0.05)], transform=ax_head.transAxes, fc=C_ACCENT, ec='none'))

        def draw_footer(fig, page_num):
            fig.text(0.5, 0.02, f"Page {page_num} • {settings.get('proj_name', 'Project')} Assembly Guide", ha='center', fontsize=8, color='#95a5a6', fontfamily=FONT_MAIN)

        # --- COVER PAGE ---
        fig = Figure(figsize=(8.27, 11.69)) # A4
        
        # Hero Image (Iso View)
        ax_iso = fig.add_axes([0.1, 0.35, 0.8, 0.45])
        ax_iso.axis('off')
        ax_iso.set_aspect('equal')
        total_h = len(final_geoms) * settings['mat_th']
        # draw_iso_shadowbox(ax_iso, w_mm, h_mm, total_h + 5) # DISABLED ON COVER PER REQUEST
        
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
        
        # Cover Header
        draw_header(fig, "ASSEMBLY GUIDE", "TopoBox Pro", 0)
        
        # Project Info Card
        fig.text(0.1, 0.28, settings.get('proj_name', 'My Project').upper(), fontsize=36, weight='bold', color=C_PRIMARY, fontfamily=FONT_MAIN)
        fig.text(0.1, 0.24, "3D Topographic Model", fontsize=16, color=C_TEXT, fontfamily=FONT_MAIN)
        
        # Stats Grid
        ax_stats = fig.add_axes([0.1, 0.08, 0.8, 0.12])
        ax_stats.axis('off')
        
        stats = [
            ("DIMENSIONS", f"{w_mm:.0f} x {h_mm:.0f} mm"),
            ("LAYERS", f"{n_layers}"),
            ("THICKNESS", f"{settings['mat_th']} mm"),
            ("TOTAL HEIGHT", f"{total_h:.1f} mm"),
            # REMOVED MASS AS REQUESTED
        ]
        
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
            ax_parts.set_aspect('equal')
            ax_parts.axis('off')
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
            ax_jig.set_aspect('equal')
            ax_jig.axis('off')
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
            ax_iso.set_aspect('equal')
            ax_iso.axis('off')
            ax_iso.set_title("3. Assembly", loc='left', fontsize=10, color=C_ACCENT, weight='bold')
            draw_iso_shadowbox(ax_iso, w_mm, h_mm, z_height + settings['mat_th'] + 5)
            
            # Draw Stack: Previous (Ghost) -> Current (Highlight)
            for stack_i in range(i + 1):
                is_current = (stack_i == i)
                z = stack_i * settings['mat_th']
                
                # Colors
                if is_current:
                    fc = C_WOOD
                    ec = C_PRIMARY # Highlight edge
                    lw = 0.8
                    alpha = 1.0
                else:
                    fc = '#f0f0f0' # Ghost
                    ec = '#dcdcdc'
                    lw = 0.3
                    alpha = 0.6
                
                for p in final_geoms[stack_i]:
                    if 'poly' in p and not p['poly'].is_empty:
                        polys = [p['poly']] if p['poly'].geom_type == 'Polygon' else list(p['poly'].geoms)
                        for sub in polys:
                            iso_P = affinity.affine_transform(sub, get_iso_transform(z))
                            ax_iso.add_patch(create_patch(iso_P, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha))
            
            ax_iso.autoscale_view()
            
            # Instructions Footer
            instructions = [
                f"1. Gather {part_count} part(s) for Layer {layer_num}.",
                "2. Place the Orange Jig frame on top of the previous layer." if jig_data else "2. No Jig required for this layer. Align visually.",
                "3. Apply glue to the previous layer where parts will sit.",
                "4. Drop parts into the Jig openings.",
                "5. Remove Jig carefully (if reusable) or leave for clamping."
            ]
             
            fig.text(0.05, 0.15, "\n".join(instructions), fontsize=10, color=C_TEXT, fontfamily='monospace', va='top')
            
            draw_footer(fig, i+2)
            pdf.savefig(fig)
    return buffer.getvalue()

def generate_zip_data(final_geoms, settings, jig_modifications_dict, current_dowels, export_fmt):
    zip_buffer = io.BytesIO()
    w_mm, h_mm = settings['box_w'], settings['box_h']
    proj_name = settings.get('proj_name', 'Project')
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, layer in enumerate(final_geoms):
            layer_num = i + 1
            jig_mods = jig_modifications_dict.get(f"L{layer_num}", [])
            jig_data = geometry_engine.generate_jig_geometry(layer, current_dowels, w_mm, h_mm, layer_num, jig_mods, conn_width=settings['jig_conn_width'], grid_spacing=settings.get('jig_grid_spacing', 20.0), fluid_smoothing=settings.get('jig_fluid', True))
            
            if export_fmt == "SVG":
                content = geometry_engine.generate_svg_string(layer, w_mm, h_mm, fill_color="none", stroke_color="red", add_background=False)
                zf.writestr(f"Components/{proj_name}_L{layer_num}_Parts.svg", content)
                if jig_data:
                    jig_content = geometry_engine.generate_svg_string([{'poly': jig_data['poly']}], w_mm, h_mm, fill_color="none", stroke_color="blue", add_background=False)
                    zf.writestr(f"Jigs/{proj_name}_L{layer_num}_Jig.svg", jig_content)

            elif export_fmt == "DXF":
                doc = ezdxf.new()
                msp = doc.modelspace()
                for p in layer:
                    geom = p['poly']
                    if geom.is_empty: continue
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                    for poly in polys:
                        if poly.is_empty: continue
                        msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'color': 1})
                        for interior in poly.interiors: msp.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'color': 1})
                dxf_stream = io.StringIO()
                doc.write(dxf_stream)
                zf.writestr(f"Components/{proj_name}_L{layer_num}_Parts.dxf", dxf_stream.getvalue())

                if jig_data:
                    doc_jig = ezdxf.new()
                    msp_jig = doc_jig.modelspace()
                    jig_poly = jig_data['poly']
                    polys = [jig_poly] if jig_poly.geom_type == 'Polygon' else list(jig_poly.geoms)
                    for poly in polys:
                        msp_jig.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'color': 5})
                        for interior in poly.interiors: msp_jig.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'color': 5})
                    dxf_stream_jig = io.StringIO()
                    doc_jig.write(dxf_stream_jig)
                    zf.writestr(f"Jigs/{proj_name}_L{layer_num}_Jig.dxf", dxf_stream_jig.getvalue())

            elif export_fmt in ["PDF", "PNG", "JPG"]:
                fig = Figure(figsize=(10, 10 * (h_mm / w_mm)))
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111)
                ax.set_xlim(0, w_mm); ax.set_ylim(0, h_mm); ax.set_aspect('equal'); ax.axis('off')
                patches = []
                for p in layer:
                    geom = p['poly']
                    if geom.is_empty: continue
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                    for poly in polys:
                        if poly.is_empty: continue
                        patches.append(MplPolygon(np.array(poly.exterior.coords), closed=True))
                        for interior in poly.interiors: patches.append(MplPolygon(np.array(interior.coords), closed=True))
                p = PatchCollection(patches, facecolor='none', edgecolor='black', linewidth=1)
                ax.add_collection(p)
                img_data = io.BytesIO()
                canvas.print_figure(img_data, format=export_fmt.lower(), bbox_inches='tight', pad_inches=0.1)
                zf.writestr(f"Components/{proj_name}_L{layer_num}_Parts.{export_fmt.lower()}", img_data.getvalue())
                
                if jig_data:
                    fig_jig = Figure(figsize=(10, 10 * (h_mm / w_mm)))
                    canvas_jig = FigureCanvasAgg(fig_jig)
                    ax_jig = fig_jig.add_subplot(111)
                    ax_jig.set_xlim(0, w_mm); ax_jig.set_ylim(0, h_mm); ax_jig.set_aspect('equal'); ax_jig.axis('off')
                    jig_poly = jig_data['poly']
                    polys = [jig_poly] if jig_poly.geom_type == 'Polygon' else list(jig_poly.geoms)
                    jig_patches = []
                    for poly in polys:
                        jig_patches.append(MplPolygon(np.array(poly.exterior.coords), closed=True))
                        for interior in poly.interiors: jig_patches.append(MplPolygon(np.array(interior.coords), closed=True))
                    pj = PatchCollection(jig_patches, facecolor='none', edgecolor='blue', linewidth=1, linestyle='--')
                    ax_jig.add_collection(pj)
                    img_data_jig = io.BytesIO()
                    canvas_jig.print_figure(img_data_jig, format=export_fmt.lower(), bbox_inches='tight', pad_inches=0.1)
                    zf.writestr(f"Jigs/{proj_name}_L{layer_num}_Jig.{export_fmt.lower()}", img_data_jig.getvalue())

    return zip_buffer.getvalue()

def generate_nested_zip(nested_components, nested_jigs, nested_comp_dims, nested_jig_dims, export_fmt, ex_wood_color, ex_edge_color):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        def write_sheets(sheets, prefix, sheet_dims):
            if not sheet_dims or sheet_dims[0] == 0 or sheet_dims[1] == 0: return
            ns_w, ns_h = sheet_dims
            for i, sheet in enumerate(sheets):
                sheet_polys = []
                # Segregate for DXF Layering logic
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
                
                if export_fmt == "SVG":
                    # Use dictionary mapping for mixed content colors
                    fill_col = {'part': ex_wood_color, 'jig': '#de2d26', 'default': ex_wood_color}
                    
                    # Note: generate_svg_string now supports dict for fill_color
                    svg = geometry_engine.generate_svg_string(sheet_polys, ns_w, ns_h, fill_color=fill_col, stroke_color=ex_edge_color, add_background=False)
                    zf.writestr(f"{prefix}_Sheet_{i+1}.svg", svg)

                elif export_fmt == "DXF":
                    doc = ezdxf.new(); msp = doc.modelspace()
                    doc.layers.new(name='PARTS', dxfattribs={'color': 1})
                    doc.layers.new(name='JIGS', dxfattribs={'color': 5})
                    
                    for p in dxf_parts:
                        polys = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        for poly in polys:
                            if poly.is_empty: continue
                            msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': 'PARTS'})
                            for interior in poly.interiors: msp.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'layer': 'PARTS'})
                    
                    for p in dxf_jigs:
                        polys = [p] if p.geom_type == 'Polygon' else list(p.geoms)
                        for poly in polys:
                            if poly.is_empty: continue
                            msp.add_lwpolyline(list(poly.exterior.coords), close=True, dxfattribs={'layer': 'JIGS'})
                            for interior in poly.interiors: msp.add_lwpolyline(list(interior.coords), close=True, dxfattribs={'layer': 'JIGS'})

                    dxf_stream = io.StringIO(); doc.write(dxf_stream)
                    zf.writestr(f"{prefix}_Sheet_{i+1}.dxf", dxf_stream.getvalue())
        
        if nested_components and nested_comp_dims: write_sheets(nested_components, "Components/Nested", nested_comp_dims)
        if nested_jigs and nested_jig_dims: write_sheets(nested_jigs, "Jigs/Nested", nested_jig_dims)
    return buf.getvalue()