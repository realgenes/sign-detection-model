#!/usr/bin/env python3
"""
Training Script for Sign Language Detection Model

This script will:
1. Create sample gesture data (if needed)
2. Load and prepare the training data
3. Train the CNN model
4. Save the trained model as 'cnn_model_keras2.h5'

Usage:
    python train_model.py
"""

import os
import cv2
import numpy as np
import pickle
import sqlite3
from glob import glob
import random
from sklearn.utils import shuffle
from cnn_tf import cnn_model, get_num_of_classes, get_image_size

def create_sample_gesture_data():
    """Create sample gesture data for demonstration purposes"""
    print("Creating sample gesture data...")
    
    # Create gesture database
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        
        # Add some sample gestures
        gestures = [
            (0, "Zero"), (1, "One"), (2, "Two"), (3, "Three"), (4, "Four"),
            (5, "Five"), (6, "Six"), (7, "Seven"), (8, "Eight"), (9, "Nine"),
            (10, "Hello"), (11, "Thank You"), (12, "Please"), (13, "Sorry"), (14, "Goodbye")
        ]
        
        for g_id, g_name in gestures:
            cmd = f"INSERT INTO gesture (g_id, g_name) VALUES ({g_id}, '{g_name}')"
            try:
                conn.execute(cmd)
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        conn.close()
    
    # Create sample images for each gesture
    image_x, image_y = get_image_size()
    
    for i in range(15):  # Create 15 gesture classes
        gesture_dir = f"gestures/{i}"
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
        
        # Create 100 sample images per gesture
        for j in range(1, 101):
            if not os.path.exists(f"{gesture_dir}/{j}.jpg"):
                # Create a simple pattern for each gesture
                img = np.random.randint(0, 255, (image_x, image_y), dtype=np.uint8)
                # Add some pattern based on gesture ID
                if i < 10:  # Numbers 0-9
                    img[10:40, 10:40] = i * 25  # Simple pattern
                else:  # Letters
                    img[15:35, 15:35] = (i-10) * 50
                
                cv2.imwrite(f"{gesture_dir}/{j}.jpg", img)
    
    print(f"Created sample data for {len(glob('gestures/*'))} gesture classes")

def prepare_training_data():
    """Load and prepare training data"""
    print("Preparing training data...")
    
    # Check if we have gesture data
    if not os.path.exists("gestures") or len(glob("gestures/*")) == 0:
        print("No gesture data found. Creating sample data...")
        create_sample_gesture_data()
    
    # Load images and labels
    images_labels = []
    images = glob("gestures/*/*.jpg")
    images.sort()
    
    print(f"Found {len(images)} images")
    
    for image in images:
        label = image[image.find(os.sep)+1: image.rfind(os.sep)]
        img = cv2.imread(image, 0)
        if img is not None:
            images_labels.append((np.array(img, dtype=np.uint8), int(label)))
    
    if len(images_labels) == 0:
        print("No valid images found!")
        return False
    
    # Shuffle the data
    images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
    images, labels = zip(*images_labels)
    
    print(f"Total samples: {len(images_labels)}")
    
    # Split into train/validation sets
    train_split = int(0.8 * len(images))
    
    train_images = images[:train_split]
    train_labels = labels[:train_split]
    
    val_images = images[train_split:]
    val_labels = labels[train_split:]
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Save the data
    with open("train_images", "wb") as f:
        pickle.dump(train_images, f)
    
    with open("train_labels", "wb") as f:
        pickle.dump(train_labels, f)
    
    with open("val_images", "wb") as f:
        pickle.dump(val_images, f)
    
    with open("val_labels", "wb") as f:
        pickle.dump(val_labels, f)
    
    return True

def train_model():
    """Train the CNN model"""
    print("Training the model...")
    
    try:
        # Import here to avoid circular import
        from cnn_tf import train
        train()
        
        if os.path.exists("cnn_model_keras2.h5"):
            print("✓ Model training completed successfully!")
            print("✓ Model saved as 'cnn_model_keras2.h5'")
            return True
        else:
            print("✗ Model file was not created")
            return False
            
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return False

def main():
    """Main training pipeline"""
    print("=== Sign Language Model Training ===")
    print()
    
    # Step 1: Prepare data
    if not prepare_training_data():
        print("Failed to prepare training data")
        return
    
    # Step 2: Train model
    if not train_model():
        print("Failed to train model")
        return
    
    print()
    print("=== Training Complete ===")
    print("You can now run 'python final.py' to use the trained model!")

if __name__ == "__main__":
    main()
