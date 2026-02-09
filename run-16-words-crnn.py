# CRNN Model for 16 Words Recognition
# Command Line: python run-16-words-crnn.py -i input_file_path

import os
import cv2
import numpy as np
import argparse
from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, Dropout, Input
import mediapipe as mp

# All 16 words the CRNN model can recognize
classes = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def pose_estimation(image, results):
    """Function which takes in image and results from mediapipe posenet and marks coordinates"""
    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(102,255,51), thickness=3, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    # Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(102,255,51), thickness=3, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    # Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    return image

def video_array_maker_opencv(pather, height=224, width=224, output_directory="./out", output_folder=None, remove_input=False):
    """
    Convert video to pose-embedded frames using OpenCV (no FFmpeg needed)
    Returns numpy array of shape (45, 224, 224, 3)
    """
    if output_folder is None:
        output_folder = os.path.split(pather)[1].split('.')[0]
    elif output_folder == "No":
        output_folder = ""
    
    # Read video with OpenCV
    cap = cv2.VideoCapture(pather)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    actualframe = len(frames)
    output_frames = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if actualframe >= 45:
            for i in range(45):
                x = round(actualframe / 45 * i)
                if x >= actualframe:
                    break
                frame = frames[x]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                output = pose_estimation(frame.copy(), results)
                output = cv2.resize(output, (width, height), interpolation=cv2.INTER_AREA)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output_frames.append(output)
        else:
            for i in range(actualframe):
                frame = frames[i]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(frame_rgb)
                output = pose_estimation(frame.copy(), results)
                output = cv2.resize(output, (width, height), interpolation=cv2.INTER_AREA)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output_frames.append(output)
            
            for i in range(45 - actualframe):
                newframe = np.zeros(shape=(height, width, 3), dtype=np.uint8)
                output_frames.append(newframe)
    
    return np.array(output_frames)

def initialize_crnn_model():
    """Initialize CRNN model architecture"""
    mobilenet = tf.keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet')
    
    model = Sequential()
    model.add(TimeDistributed(mobilenet, input_shape=(45, 224, 224, 3)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(16, activation="softmax"))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=tf.keras.metrics.Accuracy())
    
    # Try to load weights if available
    checkpoint_path = "crnn-model-v1-initial-attempt-files/models/full_model"
    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            print("✅ Loaded model weights from checkpoint")
        except:
            print("⚠️  Could not load weights, using untrained model")
    else:
        print("⚠️  Model checkpoint not found. Using untrained model architecture.")
        print("   Note: You need to train the model or provide the checkpoint file.")
    
    return model

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

if __name__ == '__main__':
    parser = ArgumentParser(description="CRNN Model for 16 Words Recognition")
    parser.add_argument("-i", "--input", dest="filename", required=True, type=validate_file,
                        help="input video file", metavar="FILE")
    args = parser.parse_args()
    
    input_video_path = args.filename
    
    print("="*70)
    print("🤖 CRNN MODEL - 16 WORDS RECOGNITION")
    print("="*70)
    print(f"\n📚 Total Words Model Can Recognize: {len(classes)}")
    print("\n📝 All 16 Recognizable Words:")
    for i, word in enumerate(classes, 1):
        print(f"   {i:2d}. {word}")
    
    print("\n" + "="*70)
    print("🔄 Loading CRNN Model...")
    print("="*70)
    
    try:
        os.makedirs("out", exist_ok=True)
    except:
        pass
    
    model = initialize_crnn_model()
    
    print("\n" + "="*70)
    print("🎬 Processing Video with Pose Detection...")
    print("="*70)
    
    # Process video through pipeline
    model_input = video_array_maker_opencv(input_video_path, output_directory="out", output_folder="No", remove_input=False)
    print(f"✅ Video processed: {model_input.shape} (45 frames, 224x224, RGB)")
    
    print("\n" + "="*70)
    print("🔮 Making Prediction...")
    print("="*70)
    
    # Predict
    prediction = model.predict(np.expand_dims(model_input, axis=0), verbose=0)
    arg_pred = np.argmax(prediction, axis=1)
    
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS FOR ALL 16 WORDS")
    print("="*70)
    print(f"\n📹 Video: {input_video_path}")
    print(f"🎬 Frames Processed: {model_input.shape[0]}")
    print("\n📈 Confidence Scores for ALL 16 Words:")
    print("-" * 70)
    
    # Sort by probability
    sorted_indices = np.argsort(prediction[0])[::-1]
    
    for idx in sorted_indices:
        word = classes[idx]
        prob = prediction[0][idx] * 100
        bar = "█" * int(prob / 2)
        marker = "✅" if idx == arg_pred[0] else "  "
        print(f"{marker} {word:15s}: {prob:6.2f}% {bar}")
    
    print("-" * 70)
    print(f"\n🎯 FINAL PREDICTION: {classes[arg_pred[0]]} ({prediction[0][arg_pred[0]]*100:.2f}% confidence)")
    print("="*70)
    
    # Top 3 predictions
    top3_indices = sorted_indices[:3]
    print("\n🏆 TOP 3 PREDICTIONS:")
    for rank, idx in enumerate(top3_indices, 1):
        word = classes[idx]
        prob = prediction[0][idx] * 100
        print(f"   {rank}. {word:15s} - {prob:6.2f}%")
    print("="*70 + "\n")

