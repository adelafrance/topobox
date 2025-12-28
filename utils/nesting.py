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
        
        # Grid Search
        # Heuristic: Bottom-Left fill
        # Check integer coordinates stepped by grid_step
        
        # Optimization: Don't check every pixel.
        y_max = int(self.sheet_h / self.grid_step)
        x_max = int(self.sheet_w / self.grid_step)
        
        for yi in range(y_max):
            y = yi * self.grid_step
            for xi in range(x_max):
                x = xi * self.grid_step
                
                for v in variants:
                    if x + v['w'] > self.sheet_w or y + v['h'] > self.sheet_h: continue
                    
                    cand_poly = affinity.translate(v['poly'], x, y)
                    
                    # Collision Check using Barriers (Already buffered by gap)
                    # We test if candidate intersects ANY barrier
                    collision = False
                    for b in self.current_sheet_barriers:
                        if b.intersects(cand_poly):
                            collision = True
                            break
                    
                    if not collision:
                        # Found a spot!
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
        # This ensures the next item stays 'gap' away
        self.current_sheet_barriers.append(final_poly.buffer(self.gap))