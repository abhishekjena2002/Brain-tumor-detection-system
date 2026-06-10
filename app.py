#%%
# ==========================================
# Core Libraries
# ==========================================
import os
import warnings
import numpy as np
import pandas as pd
# ==========================================
# Image Processing
# ==========================================
import cv2
from PIL import Image
# ==========================================
# Visualization
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================
# Scikit-Learn
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
# ==========================================
# TensorFlow / Keras
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Transfer Learning Models
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0
)

warnings.filterwarnings("ignore")
# %%
dataset_path = r"D:\my\study material\self project\Brain tumor\brain_tumor_dataset"

X = []
y = []

# Tumor Images
yes_path = os.path.join(dataset_path, "yes")

for img_name in os.listdir(yes_path):
    img_path = os.path.join(yes_path, img_name)

    img = cv2.imread(img_path)

    if img is not None:
        X.append(img)
        y.append(1)

# Normal Images
no_path = os.path.join(dataset_path, "no")

for img_name in os.listdir(no_path):
    img_path = os.path.join(no_path, img_name)

    img = cv2.imread(img_path)

    if img is not None:
        X.append(img)
        y.append(0)

print("Total Images:", len(X))
print("Total Labels:", len(y))
# %%
# EDA
#total no. of image
print("Total Images:", len(X))
print("Total Labels:", len(y))

#class distribution
df = pd.DataFrame({"Label": y})
print(df["Label"].value_counts())

# visualize class distribution 
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=y)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

#display sample MRi
plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")

plt.tight_layout()
plt.show()

# img size
for i in range(5):
    print(X[i].shape)

# analyze img distribution
heights = []
widths = []

for img in X:
    heights.append(img.shape[0])
    widths.append(img.shape[1])

print("Min Height:", min(heights))
print("Max Height:", max(heights))

print("Min Width:", min(widths))
print("Max Width:", max(widths))

# Visualize Image Size Distribution
plt.figure(figsize=(8,5))

sns.histplot(heights, bins=20)

plt.title("Image Height Distribution")
plt.show()

# check currupted img
corrupted = 0

for img in X:
    if img is None:
        corrupted += 1

print("Corrupted Images:", corrupted)

# pixel value ramge
print("Minimum Pixel Value:", np.min(X[0]))
print("Maximum Pixel Value:", np.max(X[0]))
# %%


# ==========================================
# IMPORT LIBRARIES
# ==========================================

import cv2
import numpy as np

# ==========================================
# IMAGE PREPROCESSING PIPELINE
# ==========================================

class ImagePreprocessingPipeline:

    def __init__(self, img_size=224):

        self.img_size = img_size

    def transform(self, img):

        # Resize
        img = cv2.resize(
            img,
            (self.img_size, self.img_size)
        )

        # Convert to float32
        img = img.astype(np.float32)

        # Normalize
        img = img / 255.0

        return img
    
# ==========================================
# LOAD PIPELINE
# ==========================================

from preprocessing import ImagePreprocessingPipeline

pipeline = ImagePreprocessingPipeline(
    img_size=224
)

# ==========================================
# APPLY PREPROCESSING
# ==========================================

processed_images = []

for img in X:

    processed_img = pipeline.transform(img)

    processed_images.append(processed_img)

# Convert to NumPy Array

X = np.array(processed_images)

y = np.array(y)

# ==========================================
# VERIFY PREPROCESSING
# ==========================================

print("Images Shape :", X.shape)
print("Labels Shape :", y.shape)

print("Minimum Pixel Value :", X.min())
print("Maximum Pixel Value :", X.max())

print("Data Type :", X.dtype)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels :", y_test.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"

)

train_datagen.fit(X_train)

print("\nFINAL DATASET CHECK")
print("="*40)

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels :", y_test.shape)

print("Train Min Pixel :", X_train.min())
print("Train Max Pixel :", X_train.max())

print("Test Min Pixel :", X_test.min())
print("Test Max Pixel :", X_test.max())

print("Data Type :", X_train.dtype)

# %%
#train test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels :", y_test.shape)

#%%
#data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)

train_datagen.fit(X_train)

# %%
# ==========================================
# VALIDATION
# ==========================================

from validation import validate_dataset

validation_results = validate_dataset(
    X,
    y
)
# %%
import great_expectations as gx

print(gx.__version__)
context = gx.get_context()

print(context)
# %%
import great_expectations as gx
context = gx.get_context(mode="file")
context.list_data_docs_sites()
context.build_data_docs()
context.open_data_docs()
# %%
import great_expectations as gx

context = gx.get_context(mode="file")

print(dir(context.suites))
print(dir(context.checkpoints))
# %%

from mlflow_training import *

cnn = create_cnn()

train_and_log_model(
    cnn,
    "CNN",
    X_train,
    y_train,
    X_test,
    y_test
)

mobilenet = create_mobilenet()

train_and_log_model(
    mobilenet,
    "MobileNetV2",
    X_train,
    y_train,
    X_test,
    y_test
)

resnet = create_resnet()

train_and_log_model(
    resnet,
    "ResNet50",
    X_train,
    y_train,
    X_test,
    y_test
)

efficientnet = create_efficientnet()

train_and_log_model(
    efficientnet,
    "EfficientNetB0",
    X_train,
    y_train,
    X_test,
    y_test
)

# %%
