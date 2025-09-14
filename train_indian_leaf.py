from leaf_augmentation import multi_generator, pipeline_containers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

# Build CNN
num_classes = len(pipeline_containers)
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
batch_size = 32
steps_per_epoch = 100  # adjust based on dataset
epochs = 20

model.fit(
    multi_generator(pipeline_containers, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs
)

# Save model
model.save("Indian-Leaf-CNN.h5")
print("✅ Model saved as Indian-Leaf-CNN.h5")
