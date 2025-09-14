import Augmentor
import numpy as np
import os
import glob
import random
import collections
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Locate folders (classes)
root_directory = "D:/data/*"  # Update to your dataset path
folders = [os.path.abspath(f) for f in glob.glob(root_directory) if os.path.isdir(f)]
print("Folders (classes) found: %s" % [os.path.split(x)[1] for x in folders])

# Create Augmentor pipelines
pipelines = {}
for folder in folders:
    class_name = os.path.split(folder)[1]
    print(f"Creating pipeline for {class_name}")
    pipelines[class_name] = Augmentor.Pipeline(folder)

# Augmentation settings
for pipeline in pipelines.values():
    pipeline.rotate(probability=0.75, max_left_rotation=10, max_right_rotation=10)
    pipeline.flip_left_right(probability=0.8)
    pipeline.skew(probability=0.4)
    pipeline.random_distortion(probability=0.5, grid_width=3, grid_height=7, magnitude=2)
    pipeline.crop_centre(probability=0.1, percentage_area=0.8)
    pipeline.sample(333)  # number of augmented samples per class

# Map labels to integers
integer_labels = {f'leaf{i:02}': i-1 for i in range(1,16)}

PipelineContainer = collections.namedtuple(
    'PipelineContainer',
    'label label_integer label_categorical pipeline generator'
)

pipeline_containers = []
for label, pipeline in pipelines.items():
    label_categorical = np.zeros(len(pipelines), dtype=int)
    label_categorical[integer_labels[label]] = 1
    pipeline_containers.append(
        PipelineContainer(
            label=label,
            label_integer=integer_labels[label],
            label_categorical=label_categorical,
            pipeline=pipeline,
            generator=pipeline.keras_generator(batch_size=1)
        )
    )

# Multi-generator function
def multi_generator(pipeline_containers, batch_size):
    while True:
        X = []
        y = []
        for _ in range(batch_size):
            container = random.choice(pipeline_containers)
            image_batch, _ = next(container.generator)
            image_batch = image_batch.reshape((224, 224, 3))
            X.append(image_batch)
            y.append(container.label_categorical)
        X = np.asarray(X)
        y = np.asarray(y)
        yield X, y
