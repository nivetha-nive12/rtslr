import os
import glob

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = content.replace('RTSLR', 'Signify').replace('rtslr', 'Signify')
        
        if content != new_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filepath}")
    except Exception as e:
        pass

files = glob.glob('templates/*.html') + ['app.py', 'app_fixed.py', 'README.md']
for filepath in files:
    replace_in_file(filepath)

print("Replacement complete.")
