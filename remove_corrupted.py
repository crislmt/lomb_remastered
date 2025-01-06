import os

src = r"C:\Users\Vincenzo\Thesis\project_backdooring\dataset\validation\malware"

for filename in os.listdir(src):
    filepath = os.path.join(src, filename)
    try:
        with open(filepath, 'r') as f:
            pass
    except:
        os.remove(filepath)
        print(f"Removed {filepath}")