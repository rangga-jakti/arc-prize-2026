content = open('solver/rules/transformations.py').read()
old = "    'fill_enclosed' : fill_enclosed,"
new = "    # fill_enclosed disabled - use build_interior_fill_transforms"
content = content.replace(old, new)
open('solver/rules/transformations.py', 'w').write(content)
print('Done, fill_enclosed disabled')
