import os
import glob

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for missing colors that are causing unreadable text
    replacements = {
        '#faf8f5': '#1e293b', # Light beige background -> slate-800
        '#f8f9fa': '#1e293b',
    }

    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
    
    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for filepath in glob.glob('templates/*.html'):
    replace_in_file(filepath)

print("Additional theme updates completed.")
