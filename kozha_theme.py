import os
import glob
import re

def apply_kozha_theme(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Outer Backgrounds -> #f3efe6
        bg_outers = ['#04070f', '#0a0a1a', '#0f0f1a', '#0b1121', '#1a1a2e', '#000000', '#000']
        for c in bg_outers:
            content = content.replace(c, '#f3efe6')
        content = re.sub(r'bg-\[#(04070f|0a0a1a|0f0f1a|0b1121|1a1a2e)\]', 'bg-[#f3efe6]', content)
        content = content.replace('bg-black', 'bg-[#f3efe6]')

        # 2. Cards & Elements Backgrounds -> #ffffff
        bg_cards = ['#101828', '#1e2d47', '#1e293b']
        for c in bg_cards:
            content = content.replace(c, '#ffffff')
        content = re.sub(r'bg-\[#(101828|1e2d47|1e293b)\]', 'bg-[#ffffff]', content)
        content = content.replace('rgba(255, 255, 255, 0.05)', '#ffffff')

        # 3. Main Text -> #2d2926
        text_main = ['#e8edf5', '#f1f5f9', '#ffffff', '#fff']
        for c in text_main:
            # We don't want to replace #ffffff blindly everywhere since it's the card background now,
            # but we can replace text colors
            pass
        content = re.sub(r'text-\[#(e8edf5|f1f5f9)\]', 'text-[#2d2926]', content)
        content = content.replace('color: #fff;', 'color: #2d2926;')
        content = content.replace('color: #e8edf5;', 'color: #2d2926;')

        # 4. Muted Text -> #8c8077
        content = re.sub(r'text-\[#(5a6a82|94a3b8)\]', 'text-[#8c8077]', content)
        content = content.replace('text-white/50', 'text-[#8c8077]')
        content = content.replace('text-white/60', 'text-[#8c8077]')

        # 5. Borders -> #e6dccb
        content = re.sub(r'border-\[#(1e2d47|334155)\]', 'border-[#e6dccb]', content)

        # 6. Specific overrides for the Kozha feel
        content = content.replace('text-white', 'text-[#2d2926]')

        # Also rename Signify just in case it was reverted
        content = content.replace('RTSLR', 'Signify').replace('rtslr', 'Signify')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Applied Kozha theme to {filepath}")
    except Exception as e:
        print(e)

for filepath in glob.glob('templates/*.html'):
    apply_kozha_theme(filepath)
