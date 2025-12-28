from shapely.geometry import Polygon, Point, box
from shapely import affinity
import math
import numpy as np
from scipy.signal import fftconvolve
from rasterio import features
from rasterio.transform import from_origin

class MultiSheetPacker:
    def __init__(self, sheet_w, sheet_h, gap, allow_rotation=True, grid_step=1.0):
        # Grid Step is now Pixel Resolution (mm/pixel)
        self.res = 1.0 # 1mm resolution is highly precise for laser cutting
        self.sheet_w_mm = sheet_w
        self.sheet_h_mm = sheet_h
        
        # Ensure dimensions are valid for raster
        self.w_px = max(10, int(math.ceil(sheet_w / self.res)))
        self.h_px = max(10, int(math.ceil(sheet_h / self.res)))
        
        self.gap = gap
        self.allow_rotation = allow_rotation
        
        self.sheets = [] 
        self.current_sheet_items = []
        # Raster Grid: 1 = Occupied (Keep Out), 0 = Free
        self.current_grid = np.zeros((self.h_px, self.w_px), dtype=int)

    def start_new_sheet(self):
        if self.current_sheet_items:
            self.sheets.append(self.current_sheet_items)
        self.current_sheet_items = []
        self.current_grid = np.zeros((self.h_px, self.w_px), dtype=int)

    def pack_items(self, items):
        # Sort by Bounding Box Area (Largest footprint first)
        def bbox_area(item):
            minx, miny, maxx, maxy = item['poly'].bounds
            return (maxx - minx) * (maxy - miny)
        sorted_items = sorted(items, key=bbox_area, reverse=True)
        
        for item in sorted_items:
            placed = False
            if self._place_on_current(item):
                placed = True
            
            if not placed:
                self.start_new_sheet()
                if not self._place_on_current(item):
                    # Fallback: Force center placement
                    self._add_to_sheet(item['poly'], 0, 0, item, 0, update_grid=False)

        if self.current_sheet_items:
            self.sheets.append(self.current_sheet_items)

    def _rasterize_poly(self, poly, w, h):
        # Rasterize polygon into a w x h grid
        # Use Affine identity to map (col, row) -> (x, y)
        # We assume 1px = 1 unit.
        return features.rasterize([poly], out_shape=(h, w), 
                                  transform=from_origin(0, h, self.res, self.res), 
                                  default_value=1, all_touched=True).astype(int)

    def _place_on_current(self, item_data):
        poly = item_data['poly']
        orientations = [0]
        if self.allow_rotation: orientations.extend([90, 180, 270])
        
        current_grid = self.current_grid
        if not np.any(current_grid):
            # Empty sheet? Place at 0,0 immediately (optimize)
            # Standard logic will find 0,0 anyway, but this is faster.
            pass

        best_fit = None
        # Score: tuple (y, x). Lower is better. 
        # We prefer satisfying the grid constraints FIRST.
        
        for rot in orientations:
            # 1. Rotate
            p_rot = affinity.rotate(poly, rot, origin=(0,0)) if rot != 0 else poly
            minx, miny, maxx, maxy = p_rot.bounds
            
            # Normalize to 0,0 locally
            p_norm = affinity.translate(p_rot, -minx, -miny)
            item_w = maxx - minx
            item_h = maxy - miny
            
            # 2. Rasterize Item
            # We add GAP dilation to the item mask for the CHECK.
            # This ensures we don't put it closer than GAP to impediments.
            # Actually, typically we dilate the BARRIERS.
            # But the 'current_grid' contains Occupied zones.
            # If we check Intersection(Occupied, Item), we allow 0 distance.
            # We want Distance >= Gap.
            # So checking Intersection(DilatedOccupied, Item) OR Intersection(Occupied, DilatedItem).
            # Dilating the Item is easier here since we generate it fresh.
            mask_poly = p_norm.buffer(self.gap)
            m_minx, m_miny, m_maxx, m_maxy = mask_poly.bounds
            
            # The mask must be consistent with the p_norm anchor.
            # p_norm is at 0,0. mask_poly might be at -gap, -gap.
            # We need to rasterize the mask such that (0,0) corresponds to p_norm's (0,0).
            # So we shift mask_poly by +gap?
            # Let's think relative to the top-left of the MASK IMAGE.
            
            # Mask Image Width:
            mask_px_w = int(math.ceil( (m_maxx - m_minx) / self.res )) + 1
            mask_px_h = int(math.ceil( (m_maxy - m_miny) / self.res )) + 1
            
            if mask_px_w > self.w_px or mask_px_h > self.h_px: continue

            # Rasterize mask_poly relative to its own bounds
            mask_norm = affinity.translate(mask_poly, -m_minx, -m_miny)
            
            # features.from_origin(0, height, ...) maps 0,0(px) to 0,height(coords).
            # So Y is flipped.
            # To get standard correlation:
            item_grid = features.rasterize([mask_norm], out_shape=(mask_px_h, mask_px_w),
                                         transform=from_origin(0, mask_px_h, self.res, self.res),
                                         default_value=1, all_touched=True)
            
            # Convolution
            result = fftconvolve(current_grid, item_grid[::-1, ::-1], mode='valid')
            
            # Pixels < 0.5 are valid TOP-LEFT positions for the Mask Image.
            # "valid" means the Mask fits fully inside the Sheet.
            valid_locs = np.argwhere(result < 0.5)
            
            if valid_locs.size > 0:
                # valid_locs contains [row, col].
                # Standard packing minimizes Y (Height on sheet) then X.
                # In our Raster Transform (from_origin(0, H)), Row 0 is Y=Max. Row H is Y=0.
                # So to minimize Y (Bottom packing), we want MAX ROW.
                
                # Sort valid_locs: Primary key = -row (Descending row => Bottom). Secondary = col (Left).
                # argwhere returns sorted by row ascending.
                # So we must resort.
                # We want the LAST row first?
                # valid_locs[i] = (r, c).
                # We want max(r), then min(c).
                # Key: (-r, c).
                
                # For efficiency, we just iterate or sort.
                best_idx = 0
                best_r = -1
                best_c = 999999
                
                # Manual finding of best spot (Bottom-Left)
                # Bottom means MAX row index.
                # Left means MIN col index.
                
                for i in range(len(valid_locs)):
                   r, c = valid_locs[i]
                   # We prioritize Bottom-Left.
                   # Just strictly bottom (max r). Tie-break left.
                   if r > best_r:
                       best_r = r
                       best_c = c
                   elif r == best_r:
                       if c < best_c:
                           best_c = c
                
                # We found best for this rotation.
                # Now compare with best overall.
                # Since we iterate rotations, we pick ANY valid for this rotation.
                # But we should pick the rotation that gives the absolute best pack?
                # Usually we search all rotations and pick the one with lowest Y?
                
                # Current candidate for this rotation:
                cand_y_coord = self.sheet_h_mm - (best_r * self.res) - (mask_px_h * self.res) 
                # Note: this Y is the Bottom of the MASK.
                
                if best_fit is None or cand_y_coord < best_fit[2]: # Minimize Y
                     # Calculate actual placement of the OBJECT (not the mask).
                     # Mask is padded by offset (e.g. gap).
                     # m_minx, m_miny relative to p_norm(0,0). (Usually negative).
                     # Position of Mask Top-Left on Grid: (best_c, best_r).
                     # Grid X = best_c * res.
                     # Grid Y (top) = SheetH - best_r * res.
                     
                     # Mask Left = best_c * res.
                     # Mask Top = SheetH - best_r * res.
                     # p_norm Left = Mask Left - m_minx? (m_minx is negative offset).
                     # Wait. mask_norm was translated by -m_minx.
                     # So Mask(0,0) = mask_poly(m_minx, m_maxy).
                     # We want p_norm(0,0).
                     # p_norm(0,0) is at Mask( -m_minx, ...).
                     
                     # Let's derive X:
                     # MaskLeft = best_c * res.
                     # MaskLeft accounts for m_minx.
                     # ObjectLeft = MaskLeft + m_minx. (e.g. if m_minx is -4, we add -4).
                     
                     # Check Y:
                     # MaskTop = SheetH - best_r * res.
                     # MaskTop corresponds to m_maxy (relative).
                     # ObjectTop = MaskTop + (p_maxy - m_maxy)?
                     # No. ObjectTop = MaskTop - (m_maxy - 0).
                     # ObjectY (bottom) = ObjectTop - item_h.
                     
                     # Correction Applied Below
                     pass
                     # Better:
                     # Mask Bottom = MaskTop - mask_px_h*res.
                     # Object Bottom = MaskBottom + (0 - m_miny).
                     # FIX: m_miny is negative offset (e.g. -gap).
                     # Origin Y = MaskBottomY - m_miny (e.g. 100 - (-4) = 104).
                     obj_y = (self.sheet_h_mm - (best_r * self.res) - (mask_px_h * self.res)) - m_miny
                     
                     # Same for X: MaskLeft - m_minx
                     obj_x = (best_c * self.res) - m_minx
                     
                     best_fit = (rot, obj_x, obj_y, p_norm)

        if best_fit:
            rot, x, y, p_norm = best_fit
            self._add_to_sheet(p_norm, x, y, item_data, rot)
            return True
        return False

    def _add_to_sheet(self, poly_norm, x, y, item_data, rot, update_grid=True):
        final_poly = affinity.translate(poly_norm, x, y)
        self.current_sheet_items.append({
            'x': x, 'y': y,
            'w': final_poly.bounds[2]-final_poly.bounds[0], 'h': final_poly.bounds[3]-final_poly.bounds[1],
            'data': item_data,
            'rotation': rot,
            'poly': final_poly
        })
        
        if update_grid:
            # Update the Grid with the ACTUAL object (no buffer needed for the grid itself, 
            # because we buffer the item during check? No, usually we buffer the grid).
            # Wait, if we buffer the ITEM during check, we don't need to buffer the grid.
            # Checking DilatedItem vs NormalGrid is equivalent to DilatedGrid vs NormalItem.
            # We used DilatedItem. So NormalGrid is correct.
            
            # Rasterize Normal Item
            p_rot = final_poly
            minx, miny, maxx, maxy = p_rot.bounds
            
            # We need to drop it into the global grid.
            # We rasterize it locally then paste?
            # Or rasterize freely?
            # Rasterizing freely onto 'current_grid' array using logic?
            # features.rasterize supports 'window' but maybe easier to just draw.
            
            # Create a drawing of the polygon
            # We can rasterize just this polygon into the full shape
            # shape=(h_px, w_px). transform=from_origin(0, sheet_h, res, res).
            occupancy = features.rasterize([final_poly], out_shape=(self.h_px, self.w_px),
                                         transform=from_origin(0, self.sheet_h_mm, self.res, self.res),
                                         default_value=1, all_touched=True).astype(int)
            self.current_grid = np.maximum(self.current_grid, occupancy)