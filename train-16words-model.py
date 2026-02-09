"""
Training Script for 16-Word Sign Language Recognition Model
This script trains the LSTM model on video data for all 16 words.

Dataset Structure Expected:
training-data/
├── Loud/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── They/
│   ├── video1.mp4
│   └── ...
├── Hello/
└── ... (all 16 word folders)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from helper_functions import convert_video_to_pose_embedded_np_array
import pickle
from tqdm import tqdm

# All 16 words - MUST match folder names
WORDS = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

def create_model():
    """Create 16-word LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(256, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(WORDS), activation='softmax')
    ])
    
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'top_3_accuracy']
    )
    
    return model

def load_dataset(data_dir="training-data"):
    """
    Load video dataset from folder structure
    Expected structure: data_dir/WordName/video1.mp4, video2.mp4, ...
    """
    X = []  # Features (video keypoints)
    y = []  # Labels (word indices)
    
    print("="*70)
    print("📂 Loading Dataset...")
    print("="*70)
    
    word_to_idx = {word: idx for idx, word in enumerate(WORDS)}
    
    total_videos = 0
    for word in WORDS:
        word_dir = os.path.join(data_dir, word)
        if not os.path.exists(word_dir):
            print(f"⚠️  Warning: Folder '{word}' not found in {data_dir}")
            continue
        
        video_files = [f for f in os.listdir(word_dir) if f.endswith(('.mp4', '.avi', '.mov', '.MOV', '.MP4'))]
        
        if len(video_files) == 0:
            print(f"⚠️  Warning: No videos found in '{word}' folder")
            continue
        
        print(f"\n📁 Processing '{word}': {len(video_files)} videos")
        
        for video_file in tqdm(video_files, desc=f"  {word}"):
            video_path = os.path.join(word_dir, video_file)
            try:
                # Extract keypoints from video
                keypoints = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
                
                if keypoints.shape == (45, 258):  # Validate shape
                    X.append(keypoints)
                    y.append(word_to_idx[word])
                    total_videos += 1
                else:
                    print(f"    ⚠️  Skipped {video_file}: Wrong shape {keypoints.shape}")
            except Exception as e:
                print(f"    ❌ Error processing {video_file}: {e}")
                continue
    
    if len(X) == 0:
        raise ValueError("No valid videos found! Please check your dataset structure.")
    
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(WORDS))
    
    print("\n" + "="*70)
    print("✅ Dataset Loaded Successfully!")
    print("="*70)
    print(f"📊 Total Videos: {total_videos}")
    print(f"📊 Features Shape: {X.shape}")
    print(f"📊 Labels Shape: {y_categorical.shape}")
    print(f"📊 Number of Classes: {len(WORDS)}")
    
    # Show distribution
    print("\n📈 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"   {WORDS[idx]:15s}: {count:3d} videos")
    
    return X, y_categorical

def train_model(X, y, epochs=100, batch_size=32, validation_split=0.2):
    """Train the model"""
    print("\n" + "="*70)
    print("🎓 Starting Training...")
    print("="*70)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    
    # Create model
    model = create_model()
    print("\n" + model.summary())
    
    # Callbacks
    checkpoint_path = "lstm-model-16words/best_model.hdf5"
    os.makedirs("lstm-model-16words", exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = "lstm-model-16words/final_model.hdf5"
    model.save_weights(final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")
    
    return model, history

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 16-word Sign Language Recognition Model")
    parser.add_argument("--data-dir", type=str, default="training-data",
                        help="Directory containing video folders (default: training-data)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🤖 16-WORD SIGN LANGUAGE MODEL TRAINING")
    print("="*70)
    print(f"\n📚 Words to train: {len(WORDS)}")
    for i, word in enumerate(WORDS, 1):
        print(f"   {i:2d}. {word}")
    
    print(f"\n📂 Data directory: {args.data_dir}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Load dataset
    try:
        X, y = load_dataset(args.data_dir)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n📋 Expected dataset structure:")
        print("   training-data/")
        for word in WORDS:
            print(f"   ├── {word}/")
            print(f"   │   ├── video1.mp4")
            print(f"   │   └── video2.mp4")
        exit(1)
    
    # Train model
    try:
        model, history = train_model(X, y, epochs=args.epochs, 
                                     batch_size=args.batch_size,
                                     validation_split=args.validation_split)
        print("\n" + "="*70)
        print("✅ Training Completed Successfully!")
        print("="*70)
        print(f"📁 Model saved in: lstm-model-16words/")
        print(f"   - best_model.hdf5 (best validation accuracy)")
        print(f"   - final_model.hdf5 (final epoch)")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

