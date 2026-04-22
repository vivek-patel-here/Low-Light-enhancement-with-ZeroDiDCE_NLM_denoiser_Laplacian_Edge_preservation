import os
import pyiqa

metric = pyiqa.create_metric('nima')

img_dir = "result/output"

scores = []

for file in sorted(os.listdir(img_dir)):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(img_dir, file)
        score = metric(path).item()
        print(file, ":", round(score, 4))
        scores.append(score)

print("\nAverage NIMA:", sum(scores)/len(scores))