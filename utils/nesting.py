from shapely.geometry import Polygon, Point
from shapely import affinity

class IrregularPacker:
    def __init__(self, sheet_w, sheet_h, gap, allow_rotation=True, grid_step=2.0):
        self.sheet_w = sheet_w
        self.sheet_h = sheet_h
        self.gap = gap
        self.allow_rotation = allow_rotation
        self.grid_step = grid_step
        self.sheets = [] 
        self.current_sheet_items = []
        self._add_sheet()

    def _add_sheet(self):
        if self.current_sheet_items:
            self.sheets.append(self.current_sheet_items)
        self.current_sheet_items = []
        self.placed_polys = [] # List of buffered polygons on current sheet

    def _place_item(self, width, height, item_data):
        # Note: width/height args are bounding box sizes, but we use the actual poly
        poly = item_data['poly']
        
        orientations = [0]
        if self.allow_rotation: orientations.extend([90, 180, 270])
        
        # Pre-calculate variants to save time in loop
        variants = []
        for rot in orientations:
            # Rotate
            p_rot = affinity.rotate(poly, rot, origin=(0,0)) if rot != 0 else poly
            # Normalize to (0,0)
            minx, miny, maxx, maxy = p_rot.bounds
            p_norm = affinity.translate(p_rot, -minx, -miny)
            w = maxx - minx
            h = maxy - miny
            variants.append({
                'poly': p_norm,
                'w': w, 'h': h,
                'rot': rot
            })

        # Grid Search (Bottom-Left heuristic)
        # We scan Y then X to fill from bottom-left
        # Step size determines speed vs density. 
        # Using integer range for loop, scaled by grid_step
        y_steps = int(self.sheet_h / self.grid_step)
        x_steps = int(self.sheet_w / self.grid_step)

        for yi in range(y_steps):
            y = yi * self.grid_step
            for xi in range(x_steps):
                x = xi * self.grid_step
                
                for v in variants:
                    # Check sheet bounds
                    if x + v['w'] > self.sheet_w or y + v['h'] > self.sheet_h:
                        continue
                    
                    # Translate candidate to position
                    cand_poly = affinity.translate(v['poly'], x, y)
                    
                    # Check collision with placed items (which are already buffered)
                    collision = False
                    for placed in self.placed_polys:
                        if placed.intersects(cand_poly):
                            collision = True
                            break
                    
                    if not collision:
                        # Place it!
                        self.current_sheet_items.append({
                            'x': x, 'y': y,
                            'w': v['w'], 'h': v['h'],
                            'data': item_data,
                            'rotation': v['rot']
                        })
                        # Add to placed list, buffered by gap
                        # This ensures the next part stays 'gap' away from this one
                        self.placed_polys.append(cand_poly.buffer(self.gap))
                        return True

        return False

    def finish(self):
        if self.current_sheet_items: self.sheets.append(self.current_sheet_items)