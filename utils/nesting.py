from shapely.geometry import Polygon, Point
from shapely import affinity
import math

class MultiSheetPacker:
    def __init__(self, sheet_w, sheet_h, gap, allow_rotation=True, grid_step=5.0):
        self.sheet_w = sheet_w
        self.sheet_h = sheet_h
        self.gap = gap
        self.allow_rotation = allow_rotation
        self.grid_step = grid_step
        self.sheets = [] # List of list of items
        self.current_sheet_items = []
        self.current_sheet_barriers = [] # Buffered polygons (barriers) on current sheet
        
        self.start_new_sheet()

    def start_new_sheet(self):
        if self.current_sheet_items:
            self.sheets.append(self.current_sheet_items)
        self.current_sheet_items = []
        self.current_sheet_barriers = []

    def pack_items(self, items):
        # Sort by Area (Largest first) - Best heuristic for bin packing
        sorted_items = sorted(items, key=lambda x: x['poly'].area, reverse=True)
        
        for item in sorted_items:
            placed = False
            
            # Try to place on current sheet
            if self._place_on_current(item):
                placed = True
            
            # If failed, start new sheet and try again
            if not placed:
                self.start_new_sheet()
                # If it doesn't fit on an empty new sheet, it's too big!
                if not self._place_on_current(item):
                    # Fallback: Just put it in centered (it will be oversized)
                    # We can't physical cut it, but we can't delete it.
                    self._add_to_sheet(item['poly'], 0, 0, item, 0, force=True)
        
        # Finish last sheet
        if self.current_sheet_items:
            self.sheets.append(self.current_sheet_items)

    def _place_on_current(self, item_data):
        poly = item_data['poly']
        
        orientations = [0]
        if self.allow_rotation: orientations.extend([90, 180, 270])
        
        variants = []
        for rot in orientations:
            # Rotate
            p_rot = affinity.rotate(poly, rot, origin=(0,0)) if rot != 0 else poly
            # Normalize to (0,0) bounds
            minx, miny, maxx, maxy = p_rot.bounds
            w, h = maxx - minx, maxy - miny
            variants.append({'poly': affinity.translate(p_rot, -minx, -miny), 'w': w, 'h': h, 'rot': rot})
        
    def _place_on_current(self, item_data):
        poly = item_data['poly']
        
        orientations = [0]
        if self.allow_rotation: orientations.extend([90, 180, 270])
        
        variants = []
        for rot in orientations:
            # Rotate
            p_rot = affinity.rotate(poly, rot, origin=(0,0)) if rot != 0 else poly
            # Normalize to (0,0) bounds
            minx, miny, maxx, maxy = p_rot.bounds
            w, h = maxx - minx, maxy - miny
            variants.append({'poly': affinity.translate(p_rot, -minx, -miny), 'w': w, 'h': h, 'rot': rot})
        
        # HEURISTIC OPTIMIZATION: "Candidate Points"
        # Instead of checking every 5mm (Grid Search), we only check "interesting" spots.
        # Interesting spots are: (0,0) and corners of existing placed items.
        # This reduces search space from ~20,000 points to ~50 points.
        
        candidate_points = [(0,0)] # Always try origin
        for b in self.current_sheet_barriers:
            b_minx, b_miny, b_maxx, b_maxy = b.bounds
            # Try placing to the right of existing
            if b_maxx + self.gap <= self.sheet_w:
                candidate_points.append((b_maxx + self.gap, b_miny)) # Align bottom
                candidate_points.append((b_maxx + self.gap, 0))      # Floor align
            # Try placing on top of existing
            if b_maxy + self.gap <= self.sheet_h:
                candidate_points.append((b_minx, b_maxy + self.gap)) # Align left
                candidate_points.append((0, b_maxy + self.gap))      # Wall align

        # Dedup and sort candidates by Y then X (Bottom-Left Strategy)
        # Using set for dedup requires tuples
        unique_candidates = sorted(list(set(candidate_points)), key=lambda p: (p[1], p[0]))
        
        for x, y in unique_candidates:
             for v in variants:
                if x + v['w'] > self.sheet_w or y + v['h'] > self.sheet_h: continue
                
                # Check Box First
                cand_minx, cand_miny = x, y
                cand_maxx, cand_maxy = x + v['w'], y + v['h']
                
                collision = False
                cand_poly = None 
                
                for b in self.current_sheet_barriers:
                    b_minx, b_miny, b_maxx, b_maxy = b.bounds
                    if (cand_maxx < b_minx or cand_minx > b_maxx or 
                        cand_maxy < b_miny or cand_miny > b_maxy):
                        continue 

                    if cand_poly is None:
                        cand_poly = affinity.translate(v['poly'], x, y)
                        
                    if b.intersects(cand_poly):
                        collision = True
                        break
                
                if not collision:
                    if cand_poly is None:
                            cand_poly = affinity.translate(v['poly'], x, y)
                            
                    self._add_to_sheet(v['poly'], x, y, item_data, v['rot'])
                    return True
        return False

    def _add_to_sheet(self, poly_norm, x, y, item_data, rot, force=False):
        final_poly = affinity.translate(poly_norm, x, y)
        self.current_sheet_items.append({
            'x': x, 'y': y,
            'w': final_poly.bounds[2]-final_poly.bounds[0], 'h': final_poly.bounds[3]-final_poly.bounds[1],
            'data': item_data,
            'rotation': rot,
            'poly': final_poly # Actual position
        })
        # Add barrier: The polygon buffered by 'gap'
        # OPTIMIZATION: Use low resolution buffer (resolution=2 => Octagon)
        # This keeps vertex count low while maintaining gap.
        # join_style=2 (Mitre) is faster than Round.
        self.current_sheet_barriers.append(final_poly.buffer(self.gap, resolution=2, join_style=2))