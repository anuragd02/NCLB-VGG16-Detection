# ============================================================
# REPRODUCIBLE VGG16 MODEL FOR NCLB CLASSIFICATION
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf

# -------------------------------
# 1. SET RANDOM SEED (CRITICAL)
# -------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Optional (strong reproducibility)
tf.config.experimental.enable_op_determinism()

# -------------------------------
# 2. IMPORT LIBRARIES
# -------------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error,
    explained_variance_score, r2_score
)

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 3. DATA PATHS
# -------------------------------
base_dir = "D:/AI Algorithm/Maize/Maize NCLB Disease/"
train_dir = os.path.join(base_dir, "train")
validate_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# -------------------------------
# 4. DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

validate_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# -------------------------------
# 5. DATA LOADING (WITH SEED)
# -------------------------------
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    seed=SEED
)

validate_data = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    seed=SEED
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False,
    seed=SEED
)

# -------------------------------
# 6. MODEL (VGG16 TRANSFER LEARNING)
# -------------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 7. TRAINING
# -------------------------------
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validate_data
)

# -------------------------------
# 8. TEST EVALUATION
# -------------------------------
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).ravel()
y_true = test_data.classes

# -------------------------------
# 9. CLASSIFICATION METRICS
# -------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=1)
recall = recall_score(y_true, y_pred, zero_division=1)
f1 = f1_score(y_true, y_pred, zero_division=1)

try:
    auc = roc_auc_score(y_true, y_pred_probs.ravel())
except ValueError:
    auc = 0.0

conf_matrix = confusion_matrix(y_true, y_pred)

# -------------------------------
# 10. REGRESSION METRICS
# -------------------------------
mae = mean_absolute_error(y_true, y_pred_probs.ravel())
mse = mean_squared_error(y_true, y_pred_probs.ravel())
rmse = np.sqrt(mse)
explained_var = explained_variance_score(y_true, y_pred_probs.ravel())
r2 = r2_score(y_true, y_pred_probs.ravel())
mbd = np.mean(y_pred_probs.ravel() - y_true)

# -------------------------------
# 11. PRINT RESULTS
# -------------------------------
print("\n=== Classification Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print("\n=== Regression Metrics ===")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Explained Variance: {explained_var:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"MBD: {mbd:.4f}")

# -------------------------------
# 12. SAVE MODEL
# -------------------------------
model.save("nclb_vgg16.keras")
print("Model saved successfully!")

# -------------------------------
# 13. LEARNING CURVES
# -------------------------------
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.figure()
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy vs Epochs')
plt.show()

plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Loss vs Epochs')
plt.show()

# -------------------------------
# 14. SAVE METRICS TABLE
# -------------------------------
df = pd.DataFrame({
    'Epoch': epochs,
    'Train Accuracy': train_acc,
    'Validation Accuracy': val_acc
})

df.to_csv("metrics_per_epoch.csv", index=False)
print("Metrics saved!")
