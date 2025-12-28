from shapely.geometry import box
from utils.nesting import MultiSheetPacker

def test_packing():
    print("Testing MultiSheetPacker...")
    
    # 1. Setup: Small Sheet (100x100)
    packer = MultiSheetPacker(sheet_w=100, sheet_h=100, gap=2, allow_rotation=True, grid_step=5.0)
    
    # 2. Create Items: 3 items of 60x60
    # One 60x60 fits per 100x100 sheet (if gap is small). 
    # Two would require 120 width. 
    # So 3 items should result in 3 sheets.
    items = [
        {'poly': box(0, 0, 60, 60), 'id': 1},
        {'poly': box(0, 0, 60, 60), 'id': 2},
        {'poly': box(0, 0, 60, 60), 'id': 3},
    ]
    
    # 3. Pack
    packer.pack_items(items)
    
    # 4. Verify
    n_sheets = len(packer.sheets)
    print(f"Packed into {n_sheets} sheets.")
    
    if n_sheets == 3:
        print("✅ SUCCESS: Correctly overflowed to 3 sheets.")
    else:
        print(f"❌ FAILURE: Expected 3 sheets, got {n_sheets}")

    # Test 2: Rotation
    # Item is 120x40. Sheet is 100x100.
    # Should fit if rotated (40x120 fits? No wait. 120 > 100).
    # Let's try Item 90x40. Sheet 50x100.
    # W=90 > SheetW=50. Must rotate to H=90, W=40.
    print("\nTesting Rotation...")
    packer2 = MultiSheetPacker(sheet_w=50, sheet_h=100, gap=1, allow_rotation=True)
    item_long = {'poly': box(0, 0, 90, 40), 'id': 'long'} # 90 wide, 40 high
    packer2.pack_items([item_long])
    
    if len(packer2.sheets) == 1 and len(packer2.sheets[0]) == 1:
        placed = packer2.sheets[0][0]
        rot = placed.get('rotation', 0)
        print(f"✅ SUCCESS: Partial fit with rotation {rot}.")
    else:
         print(f"❌ FAILURE: Failed to fit rotated item. Sheets: {len(packer2.sheets)}")

if __name__ == "__main__":
    test_packing()
