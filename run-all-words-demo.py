# Demo script to show all words the model can recognize
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from helper_functions import convert_video_to_pose_embedded_np_array

# All words the model can recognize
actions = np.array(["Hello", "How are you", "thank you"])

def initialize_model():
    """ Initializes lstm model and loads the trained model weight  """
    model = Sequential()
    model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(45,258)))
    model.add(LSTM(128,return_sequences=True, activation = 'relu'))
    model.add(LSTM(256,return_sequences=True,activation="relu"))
    model.add(LSTM(64, return_sequences = False,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))
    
    model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    model.load_weights(r"lstm-model\170-0.83.hdf5")
    return model

print("="*70)
print("🤖 INDIAN SIGN LANGUAGE RECOGNITION - ALL WORDS DEMO")
print("="*70)
print(f"\n📚 Total Words Model Can Recognize: {len(actions)}")
print("\n📝 All Recognizable Words:")
for i, word in enumerate(actions, 1):
    print(f"   {i}. {word}")

# Load model
print("\n" + "="*70)
print("🔄 Loading ML Model...")
print("="*70)
model = initialize_model()
print("✅ Model loaded successfully!")

# Test with available video
video_path = "crnn-model-v1-initial-attempt-files\\input.mp4"
if os.path.exists(video_path):
    print("\n" + "="*70)
    print("🎬 Processing Video...")
    print("="*70)
    
    out_np_array = convert_video_to_pose_embedded_np_array(video_path, remove_input=False)
    prediction = model.predict(np.expand_dims(out_np_array, axis=0))
    arg_pred = np.argmax(prediction, axis=1)
    
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS FOR ALL WORDS")
    print("="*70)
    print(f"\n📹 Video: {video_path}")
    print(f"🎬 Frames Processed: {len(out_np_array)}")
    print("\n📈 Confidence Scores for ALL Words:")
    print("-" * 70)
    
    # Sort by probability for better visualization
    sorted_indices = np.argsort(prediction[0])[::-1]
    
    for idx in sorted_indices:
        word = actions[idx]
        prob = prediction[0][idx] * 100
        bar = "█" * int(prob / 1.5)
        marker = "✅" if idx == arg_pred[0] else "  "
        print(f"{marker} {word:15s}: {prob:6.2f}% {bar}")
    
    print("-" * 70)
    print(f"\n🎯 FINAL PREDICTION: {actions[arg_pred[0]]} ({prediction[0][arg_pred[0]]*100:.2f}% confidence)")
    print("="*70)
    
    # Show all words summary
    print("\n" + "="*70)
    print("📋 SUMMARY - ALL WORDS MODEL CAN RECOGNIZE")
    print("="*70)
    for i, word in enumerate(actions, 1):
        prob = prediction[0][i-1] * 100
        status = "✅ PREDICTED" if i-1 == arg_pred[0] else "  "
        print(f"{status} {i}. {word:15s} - Confidence: {prob:5.2f}%")
    print("="*70 + "\n")
else:
    print(f"\n❌ Video not found: {video_path}")
    print("\n📋 Model is ready to recognize these words:")
    for i, word in enumerate(actions, 1):
        print(f"   {i}. {word}")

