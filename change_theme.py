import os
import glob

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for Dark Blue Theme
    replacements = {
        '#f3efe6': '#0f172a', # Light bg to Dark blue bg
        '#2d2926': '#f1f5f9', # Dark text to Light text
        '#ffffff': '#1e293b', # White card to Slate 800
        'bg-white': 'bg-[#1e293b]', # Tailwind white bg to Slate 800
        'bg-black': 'bg-[#0f172a]', # Tailwind black to Dark blue bg
        '#e6dccb': '#334155', # Border color to Slate 700
        '#8c8077': '#94a3b8', # Muted text to Slate 400
        '#04070f': '#0f172a', # In change-mode.html change dark to dark blue
    }

    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    # Also handle some rgba replacements where possible
    # rgba(255, 255, 255, 0.95) for nav glass -> rgba(30, 41, 59, 0.95)
    new_content = new_content.replace('rgba(255, 255, 255, 0.95)', 'rgba(30, 41, 59, 0.95)')
    
    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for filepath in glob.glob('templates/*.html'):
    replace_in_file(filepath)

print("Theme updated to dark blue.")
