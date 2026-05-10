# Disable transforms yang 0 correct di engine
content = open('solver/search/engine.py').read()
# Tambah blacklist
blacklist = """
TRANSFORM_BLACKLIST = {
    'remove_smaller_objects', 'remove_larger_objects',
    'fill_interior_auto', 'fill_interior_match_border',
    'complete_d_symmetry', 'complete_h_symmetry', 'complete_v_symmetry',
    'draw_rectangle_border', 'remove_minority_color', 'keep_most_common',
    'draw_outline', 'crop_to_content',
}
"""
# Insert after imports
old = 'def build_color_transforms'
new = blacklist + '\ndef build_color_transforms'
content = content.replace(old, new, 1)
# Filter results
old2 = '    results.sort(key=lambda x: x'
new2 = '    results = [r for r in results if r["transform"] not in TRANSFORM_BLACKLIST]\n    results.sort(key=lambda x: x'
content = content.replace(old2, new2, 1)
open('solver/search/engine.py', 'w').write(content)
print('Done')
