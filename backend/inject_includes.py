"""Inject components.css and theme.js into all HTML files."""
import os, re

base = r'd:\AI_ATTENDENCE SYSTEM\frontend'
html_files = []
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.html'):
            html_files.append(os.path.join(root, f))

for fp in html_files:
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changed = False
    
    # Add components.css if not already there
    if 'components.css' not in content:
        css_pattern = r'(<link\s+rel="stylesheet"\s+href="[^"]*\.css"[^>]*>)'
        matches = list(re.finditer(css_pattern, content))
        if matches:
            last_css = matches[-1]
            rel = os.path.relpath(os.path.join(base, 'css'), os.path.dirname(fp)).replace('\\', '/')
            insert_text = '\n  <link rel="stylesheet" href="' + rel + '/components.css">'
            content = content[:last_css.end()] + insert_text + content[last_css.end():]
            changed = True
    
    # Add theme.js if not already there
    if 'theme.js' not in content:
        rel_js = os.path.relpath(os.path.join(base, 'js'), os.path.dirname(fp)).replace('\\', '/')
        theme_tag = '  <script src="' + rel_js + '/theme.js"></script>\n'
        content = content.replace('</head>', theme_tag + '</head>')
        changed = True
    
    if changed:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Updated: ' + os.path.basename(fp))
    else:
        print('Skipped: ' + os.path.basename(fp))
