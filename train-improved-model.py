"""
Improved Training Script with Data Augmentation and Better Architecture
This will significantly improve model accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from helper_functions import convert_video_to_pose_embedded_np_array
import random

def detect_available_words(data_dir="training-data"):
    """Detect which words have videos in the dataset"""
    words = []
    if not os.path.exists(data_dir):
        return words
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            videos = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if len(videos) > 0:
                words.append(item)
    
    return sorted(words)

def augment_keypoints(keypoints, noise_factor=0.02):
    """Add noise augmentation to keypoints to increase dataset"""
    noise = np.random.normal(0, noise_factor, keypoints.shape)
    augmented = keypoints + noise
    return augmented

def create_improved_model(num_classes):
    """Create improved LSTM model with better architecture"""
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=(45, 258),
             kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(256, return_sequences=True, activation='tanh',
             kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(128, return_sequences=True, activation='tanh',
             kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False, activation='tanh',
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def load_dataset_with_augmentation(data_dir="training-data", augment_times=3):
    """Load dataset with augmentation to increase size"""
    X = []
    y = []
    
    words = detect_available_words(data_dir)
    
    if len(words) == 0:
        raise ValueError("No words with videos found!")
    
    print("="*70)
    print("Loading Dataset with Augmentation...")
    print("="*70)
    print(f"\nFound {len(words)} words with data:")
    for i, word in enumerate(words, 1):
        print(f"   {i}. {word}")
    
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    total_videos = 0
    total_augmented = 0
    
    for word in words:
        word_dir = os.path.join(data_dir, word)
        video_files = [f for f in os.listdir(word_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.MOV', '.MP4', '.mkv'))]
        
        print(f"\nProcessing '{word}': {len(video_files)} videos")
        
        for video_file in video_files:
            video_path = os.path.join(word_dir, video_file)
            try:
                print(f"    Loading: {video_file}")
                keypoints = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
                
                if keypoints.shape == (45, 258):
                    # Add original
                    X.append(keypoints)
                    y.append(word_to_idx[word])
                    total_videos += 1
                    
                    # Add augmented versions
                    for _ in range(augment_times):
                        augmented = augment_keypoints(keypoints)
                        X.append(augmented)
                        y.append(word_to_idx[word])
                        total_augmented += 1
                    
                    print(f"    ✅ Success: {video_file} (+{augment_times} augmented)")
                else:
                    print(f"    ⚠️  Skipped {video_file}: Wrong shape {keypoints.shape}")
            except Exception as e:
                print(f"    ❌ Error processing {video_file}: {str(e)[:100]}")
                continue
    
    if len(X) == 0:
        raise ValueError("No valid videos processed!")
    
    X = np.array(X)
    y = np.array(y)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(words))
    
    print("\n" + "="*70)
    print("Dataset Loaded with Augmentation!")
    print("="*70)
    print(f"Original Videos: {total_videos}")
    print(f"Augmented Samples: {total_augmented}")
    print(f"Total Samples: {len(X)}")
    print(f"Features Shape: {X.shape}")
    print(f"Number of Classes: {len(words)}")
    
    print("\nClass Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"   {words[idx]:15s}: {count:3d} samples")
    
    return X, y_categorical, words

def train_improved_model(X, y, words, epochs=200, batch_size=16, validation_split=0.2):
    """Train improved model"""
    print("\n" + "="*70)
    print("Starting Improved Training...")
    print("="*70)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create improved model
    model = create_improved_model(len(words))
    print("\n" + "="*70)
    print("Improved Model Architecture:")
    print("="*70)
    model.summary()
    
    # Enhanced callbacks
    model_dir = "lstm-model-improved"
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, "best_model.hdf5")
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining started with improved architecture...")
    print("="*70)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and word list
    final_model_path = os.path.join(model_dir, "final_model.hdf5")
    model.save_weights(final_model_path)
    
    words_file = os.path.join(model_dir, "words.txt")
    with open(words_file, 'w') as f:
        for word in words:
            f.write(f"{word}\n")
    
    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print(f"Model saved in: {model_dir}/")
    print(f"   - best_model.hdf5 (best validation accuracy)")
    print(f"   - final_model.hdf5 (final epoch)")
    print(f"   - words.txt (list of trained words)")
    print(f"\nTrained on {len(words)} words: {', '.join(words)}")
    
    # Print final metrics
    final_train_acc = history.history['categorical_accuracy'][-1]
    final_val_acc = history.history['val_categorical_accuracy'][-1]
    best_val_acc = max(history.history['val_categorical_accuracy'])
    
    print(f"\nFinal Training Accuracy: {final_train_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    return model, history, words

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Improved Sign Language Model")
    parser.add_argument("--data-dir", type=str, default="training-data",
                        help="Directory containing video folders")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--augment", type=int, default=4,
                        help="Number of augmented samples per video")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("IMPROVED SIGN LANGUAGE MODEL TRAINING")
    print("="*70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Augmentation: {args.augment}x per video")
    print("\nImprovements:")
    print("  ✅ Data augmentation (increases dataset size)")
    print("  ✅ Better architecture (deeper, more regularization)")
    print("  ✅ Dropout and BatchNormalization")
    print("  ✅ L2 regularization")
    print("  ✅ Better optimizer settings")
    
    try:
        # Load dataset with augmentation
        X, y, words = load_dataset_with_augmentation(
            args.data_dir, 
            augment_times=args.augment
        )
        
        # Train improved model
        model, history, trained_words = train_improved_model(
            X, y, words,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
        
        print("\n" + "="*70)
        print("SUCCESS! Improved model is ready!")
        print("="*70)
        print(f"\nTo use the improved model, update realtime-detection.py to load:")
        print(f"   Model: lstm-model-improved/best_model.hdf5")
        print(f"   Words: {trained_words}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

