import os

rainy_dir = "data/Train/Derain/degraded"
output_txt = "rainTrain.txt"

file_list = [
    f for f in os.listdir(rainy_dir) if os.path.isfile(os.path.join(rainy_dir, f))
]

with open(output_txt, "w") as f:
    for filename in sorted(file_list):
        f.write(f"{rainy_dir}/{filename}\n")
