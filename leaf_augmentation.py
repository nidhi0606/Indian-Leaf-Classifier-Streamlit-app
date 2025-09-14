import Augmentor
import glob
import os

# Dataset path
root_directory = "dataset/*"

# Get all class folders
folders = [f for f in glob.glob(root_directory) if os.path.isdir(f)]
pipelines = {}

# Create Augmentor pipelines
for folder in folders:
    class_name = os.path.split(folder)[1]
    pipelines[class_name] = Augmentor.Pipeline(folder)
    print(f"Pipeline created for: {class_name}")

# Apply augmentations
for pipeline in pipelines.values():
    pipeline.rotate(probability=0.75, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.8)
    pipeline.skew(probability=0.4)
    pipeline.random_distortion(probability=0.5, grid_width=3, grid_height=7, magnitude=2)
    pipeline.crop_centre(probability=0.1, percentage_area=0.8)
    pipeline.sample(50)

print("Augmentation done!")
