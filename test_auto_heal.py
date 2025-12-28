import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.ops import unary_union
from utils.geometry_engine import auto_heal_layer


# Mock Settings
settings = {
    'min_feature_width': 3.0,
    'box_w': 100.0,
    'box_h': 100.0,
}

def test_auto_heal_integration():
    print("Testing Auto-Heal Logic...")
    
    # 1. Create a synthetic layer roughly mimicking a real scenario
    # - A solid block (safe)
    # - A thin protrusion (needs heal)
    # - A tiny speck (noise, should be deleted)
    
    safe_block = box(10, 10, 40, 40) # 30x30mm
    thin_strip = box(40, 24.5, 60, 25.5) # 1mm wide strip connected to block
    tiny_noise = box(80, 80, 80.5, 80.5) # 0.5x0.5mm isolated speck
    
    # Combine into a "Raw Layer" list format expected by engine
    # raw_layer_geoms is a list of dicts [{'poly':, 'type': 'island'}]
    
    raw_layer = [
        {'poly': safe_block, 'type': 'island'},
        {'poly': thin_strip, 'type': 'island'},
        {'poly': tiny_noise, 'type': 'island'}
    ]
    
    print(f"Input: {len(raw_layer)} polys. Thin strip area ~20. Noise area 0.25.")
    
    # 2. Mock Terrain
    # Flat terrain for simplicity, or slope? Let's use slope to verify thicken works.
    w, h = 200, 200
    x = np.linspace(0, 100, w)
    y = np.linspace(0, 100, h)
    xv, yv = np.meshgrid(x, y)
    smooth_data = 60.0 - np.abs(xv - 50.0) * 2.0 # Ridge at x=50
    
    layer_idx = 1
    m_per_layer = 1.0
    min_elev = 0.0
    
    # Run Auto-Heal
    healed_layer, mods = auto_heal_layer(raw_layer, smooth_data, layer_idx, m_per_layer, min_elev, settings)
    
    # CHECKS
    
    # 1. Noise should be gone
    # The healed_layer is a list of classified polys.
    total_area = sum(p['poly'].area for p in healed_layer)
    print(f"Total Healed Area: {total_area:.2f}")
    
    # Safe block (900) + Thickened Strip (>20)
    # Noise (0.25) should be gone.
    
    # Check if tiny noise is present
    has_noise = any(p['poly'].intersects(box(79, 79, 81, 81)) for p in healed_layer)
    assert not has_noise, "Noise was not filtered!"
    
    # 2. Thin strip should be thickened
    # Check bounds of the area where the strip was
    strip_zone = box(40, 20, 60, 30)
    healed_geom = unary_union([p['poly'] for p in healed_layer])
    strip_part = healed_geom.intersection(strip_zone)
    
    print(f"Strip Part Area: {strip_part.area:.2f}")
    assert strip_part.area > 30.0, "Strip passed through but wasn't thickened enough!"
    
    # 3. Modifications should be tracked
    print(f"Modifications tracked: {len(mods)}")
    assert len(mods) > 0, "No modifications reported!"
    assert mods[0]['area'] > 0, "Modification has no area!"
    
    print("✅ Test Passed: Auto-Heal thickened the strip, removed noise, and reported mods.")

if __name__ == "__main__":
    test_auto_heal_integration()
