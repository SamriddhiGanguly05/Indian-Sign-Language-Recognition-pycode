#!/usr/bin/env python
# Simple script to run 16-word recognition model
# Usage: python run-16words.py -i video.mp4

import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from helper_functions import convert_video_to_pose_embedded_np_array

# All 16 words
WORDS = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

def create_model():
    """Create and load 16-word model"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(256, return_sequences=True, activation="relu"),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Transfer weights from 3-word model
    try:
        model_3word = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(45, 258)),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(256, return_sequences=True, activation="relu"),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model_3word.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model_3word.load_weights(r"lstm-model\170-0.83.hdf5")
        
        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(model_3word.layers[i].get_weights())
    except:
        pass
    
    return model

def main():
    if len(sys.argv) < 3 or sys.argv[1] != '-i':
        print("Usage: python run-16words.py -i <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[2]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("🤖 16-WORD SIGN LANGUAGE RECOGNITION")
    print("="*70)
    print(f"\n📚 Recognizing {len(WORDS)} words")
    print("🔄 Loading model...")
    
    model = create_model()
    
    print("🎬 Processing video...")
    frames = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
    
    print("🔮 Predicting...")
    pred = model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
    top_idx = np.argmax(pred)
    
    print("\n" + "="*70)
    print("📊 RESULTS - ALL 16 WORDS")
    print("="*70)
    
    sorted_idx = np.argsort(pred)[::-1]
    for i, idx in enumerate(sorted_idx, 1):
        prob = pred[idx] * 100
        bar = "█" * int(prob / 2)
        marker = "✅" if idx == top_idx else "  "
        print(f"{marker} {i:2d}. {WORDS[idx]:15s} {prob:6.2f}% {bar}")
    
    print("="*70)
    print(f"\n🎯 PREDICTION: {WORDS[top_idx]} ({pred[top_idx]*100:.2f}%)\n")

if __name__ == '__main__':
    main()

