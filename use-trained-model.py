"""
Use the trained model for predictions
This script loads the trained model and makes predictions on videos
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from helper_functions import convert_video_to_pose_embedded_np_array
import sys

def load_trained_model(model_dir="lstm-model-trained"):
    """Load the trained model and word list"""
    # Load word list
    words_file = os.path.join(model_dir, "words.txt")
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Words file not found: {words_file}")
    
    with open(words_file, 'r') as f:
        words = [line.strip() for line in f.readlines()]
    
    # Create model architecture
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(256, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(words), activation='softmax')
    ])
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Load weights
    model_path = os.path.join(model_dir, "best_model.hdf5")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.hdf5")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in {model_dir}")
    
    model.load_weights(model_path)
    
    return model, words

def predict_video(video_path, model, words):
    """Predict sign language word from video"""
    # Process video
    keypoints = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
    
    # Predict
    prediction = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
    top_idx = np.argmax(prediction)
    
    return prediction, top_idx, words

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python use-trained-model.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("="*70)
    print("🤖 LOADING TRAINED MODEL")
    print("="*70)
    
    try:
        model, words = load_trained_model()
        print(f"✅ Model loaded!")
        print(f"📚 Trained on {len(words)} words: {', '.join(words)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure training is complete first!")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🎬 Processing Video...")
    print("="*70)
    
    try:
        pred, top_idx, words_list = predict_video(video_path, model, words)
        
        print("\n" + "="*70)
        print("📊 PREDICTION RESULTS")
        print("="*70)
        print(f"\n📹 Video: {video_path}")
        
        print("\n📈 Confidence Scores:")
        sorted_idx = np.argsort(pred)[::-1]
        for idx in sorted_idx:
            word = words_list[idx]
            prob = pred[idx] * 100
            bar = "█" * int(prob / 2)
            marker = "✅" if idx == top_idx else "  "
            print(f"{marker} {word:15s}: {prob:6.2f}% {bar}")
        
        print("\n" + "="*70)
        print(f"🎯 PREDICTION: {words_list[top_idx]} ({pred[top_idx]*100:.2f}% confidence)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()

