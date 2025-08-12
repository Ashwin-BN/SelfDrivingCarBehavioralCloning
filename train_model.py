import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# Image Augmentation
# ==========================================================
def apply_random_flip(image, steering_angle):
    """
    Apply a random horizontal flip with a 50% probability.
    Flipping horizontally requires inverting the steering angle.
    """
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

# ==========================================================
# Data Loading and Balancing
# ==========================================================
def load_and_balance_data(csv_path):
    """
    Load driving log CSV, fix image paths, filter missing files,
    and balance dataset by reducing the over-represented 'straight' driving cases.
    Also shows and saves histograms of steering angle distribution.
    """
    # Load dataset
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    # Normalize image paths
    def normalize_path(p):
        return os.path.join('IMG', os.path.basename(p.strip().replace("\\", "/")))

    # Collect valid samples (center image & steering)
    valid_samples = []
    for _, row in df.iterrows():
        img_path = normalize_path(row['center'])
        angle = row['steering']
        if os.path.exists(img_path):
            valid_samples.append((img_path, angle))

    df = pd.DataFrame(valid_samples, columns=['image', 'steering'])
    df = df[df['steering'].notnull()]

    # Plot and save histogram before balancing
    plt.figure(figsize=(8, 4))
    plt.hist(df['steering'], bins=25, color='skyblue', edgecolor='black')
    plt.title("Steering Distribution (Before Balancing)")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('steering_before_balancing.png')
    plt.show()

    # Balance data: keep all turns, downsample straight driving
    turning_samples = df[abs(df['steering']) > 0.05]
    straight_samples = df[abs(df['steering']) <= 0.05].sample(frac=0.2, random_state=42)
    df_balanced = pd.concat([turning_samples, straight_samples]).sample(frac=1.0).reset_index(drop=True)

    # Plot and save histogram after balancing
    plt.figure(figsize=(8, 4))
    plt.hist(df_balanced['steering'], bins=25, color='lightgreen', edgecolor='black')
    plt.title("Steering Distribution (After Balancing)")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('steering_after_balancing.png')
    plt.show()

    return df_balanced['image'].values, df_balanced['steering'].values


# ==========================================================
# Image Preprocessing
# ==========================================================
def preprocess_image(image):
    """
    Crop irrelevant regions, convert color to YUV,
    apply Gaussian blur, resize to NVIDIA model input size,
    and normalize pixel values.
    """
    image = image[60:135, :, :]  # Remove sky and car hood
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0
    return image

# ==========================================================
# Batch Loader
# ==========================================================
def load_image_batch(paths, angles, augment=False):
    """
    Load a batch of images and corresponding steering angles from disk,
    apply augmentation if specified, and preprocess images.
    """
    images, labels = [], []
    for i in range(len(paths)):
        image_path = paths[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angle = angles[i]

        if augment:
            image, angle = apply_random_flip(image, angle)

        image = preprocess_image(image)
        images.append(image)
        labels.append(angle)

    return np.array(images), np.array(labels)

# ==========================================================
# Model Architecture
# ==========================================================
def build_nvidia_model():
    """
    Create NVIDIA's convolutional neural network architecture
    for behavioral cloning in self-driving applications.
    """
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=MeanSquaredError())
    return model

# ==========================================================
# Model Training
# ==========================================================
def train_model():
    """
    Load dataset, preprocess, train NVIDIA CNN model with early stopping,
    and save trained model to disk. Also plots training performance.
    """
    csv_file = 'driving_log.csv'
    img_paths, angles = load_and_balance_data(csv_file)

    # Train/Validation Split
    train_paths, val_paths, train_angles, val_angles = train_test_split(
        img_paths, angles, test_size=0.2, random_state=42
    )

    # Load data batches
    X_train, y_train = load_image_batch(train_paths, train_angles, augment=True)
    X_val, y_val = load_image_batch(val_paths, val_angles, augment=False)

    # Build and train model
    model = build_nvidia_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=64,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[early_stopping]
    )

    # Save model
    model.save('model.h5')
    print("Training complete. Model saved as 'model.h5'.")

    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_plot.png')
    plt.show()

# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == '__main__':
    train_model()
