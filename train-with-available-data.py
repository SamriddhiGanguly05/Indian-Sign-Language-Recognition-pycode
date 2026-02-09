"""
Flexible Training Script - Trains on whatever words are available in the dataset
This will automatically detect which words have data and train accordingly
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
# from tqdm import tqdm  # Disabled for Windows compatibility

def detect_available_words(data_dir="training-data"):
    """Detect which words have videos in the dataset"""
    words = []
    if not os.path.exists(data_dir):
        return words
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if folder has videos
            videos = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if len(videos) > 0:
                words.append(item)
    
    return sorted(words)

def create_model(num_classes):
    """Create LSTM model for specified number of classes"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(256, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

def load_dataset(data_dir="training-data"):
    """Load video dataset from available folders"""
    X = []
    y = []
    
    words = detect_available_words(data_dir)
    
    if len(words) == 0:
        raise ValueError("No words with videos found!")
    
    print("="*70)
    print("📂 Loading Dataset...")
    print("="*70)
    print(f"\n📚 Found {len(words)} words with data:")
    for i, word in enumerate(words, 1):
        print(f"   {i}. {word}")
    
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    total_videos = 0
    
    for word in words:
        word_dir = os.path.join(data_dir, word)
        video_files = [f for f in os.listdir(word_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.MOV', '.MP4', '.mkv'))]
        
        print(f"\n📁 Processing '{word}': {len(video_files)} videos")
        
        for video_file in video_files:
            video_path = os.path.join(word_dir, video_file)
            try:
                print(f"    Processing: {video_file}")
                keypoints = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
                
                if keypoints.shape == (45, 258):
                    X.append(keypoints)
                    y.append(word_to_idx[word])
                    total_videos += 1
                    print(f"    ✅ Success: {video_file}")
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
    print("✅ Dataset Loaded!")
    print("="*70)
    print(f"📊 Total Videos: {total_videos}")
    print(f"📊 Features Shape: {X.shape}")
    print(f"📊 Number of Classes: {len(words)}")
    
    print("\n📈 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"   {words[idx]:15s}: {count:3d} videos")
    
    return X, y_categorical, words

def train_model(X, y, words, epochs=100, batch_size=16, validation_split=0.2):
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
    model = create_model(len(words))
    print("\n" + "="*70)
    print("📐 Model Architecture:")
    print("="*70)
    model.summary()
    
    # Callbacks
    model_dir = "lstm-model-trained"
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, "best_model.hdf5")
    
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Training started...")
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
    
    # Save word list
    words_file = os.path.join(model_dir, "words.txt")
    with open(words_file, 'w') as f:
        for word in words:
            f.write(f"{word}\n")
    
    print("\n" + "="*70)
    print("✅ Training Completed!")
    print("="*70)
    print(f"📁 Model saved in: {model_dir}/")
    print(f"   - best_model.hdf5 (best validation accuracy)")
    print(f"   - final_model.hdf5 (final epoch)")
    print(f"   - words.txt (list of trained words)")
    print(f"\n📚 Trained on {len(words)} words: {', '.join(words)}")
    
    return model, history, words

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sign Language Model with Available Data")
    parser.add_argument("--data-dir", type=str, default="training-data",
                        help="Directory containing video folders")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIGN LANGUAGE MODEL TRAINING")
    print("="*70)
    print(f"\n📂 Data directory: {args.data_dir}")
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    try:
        # Load dataset
        X, y, words = load_dataset(args.data_dir)
        
        # Train model
        model, history, trained_words = train_model(
            X, y, words,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
        
        print("\n" + "="*70)
        print("🎉 SUCCESS! Model is ready to use!")
        print("="*70)
        print(f"\n💡 To use the trained model, update your scripts to load:")
        print(f"   Model: lstm-model-trained/best_model.hdf5")
        print(f"   Words: {trained_words}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

