content = open('solver/search/engine.py').read()
# Remove blacklist
import re
content = re.sub(r'TRANSFORM_BLACKLIST = \{[^}]+\}\n', '', content)
content = content.replace(
    '    results = [r for r in results if r["transform"] not in TRANSFORM_BLACKLIST]\n    results.sort(key=lambda x: x',
    '    results.sort(key=lambda x: x'
)
open('solver/search/engine.py', 'w').write(content)
print('Done - blacklist removed')
