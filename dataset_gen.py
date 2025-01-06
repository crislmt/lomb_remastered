import os
import shutil
import random

labels = ['malware', 'goodware']
categories = {'training': 22000, 'validation': 3000}

src = r"C:\Users\Vincenzo\Thesis\looking-out-my-backdoor-code\dnn-backdooring"
out = r"C:\Users\Vincenzo\Thesis\project_backdooring\dataset"

for label in labels:
    all_files = os.listdir(os.path.join(src, label))
    random.shuffle(all_files)
    all_files = all_files[:25000]
    for category in categories:
        for file in all_files[:categories[category]]:
            shutil.move(os.path.join(src, label, file), os.path.join(out, category, label, file))
            all_files.remove(file)


