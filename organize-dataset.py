"""
Organize the Greetings dataset for training
This script will:
1. Map folder names to our 16-word model
2. Copy/organize videos into training-data structure
3. Handle folder name variations
"""

import os
import shutil
from pathlib import Path

# Mapping from dataset folder names to our model words
FOLDER_MAPPING = {
    '48. Hello': 'Hello',
    '49. How are you': 'How are you',
    '50. Alright': 'Alright',
    '51. Good Morning': 'Good Morning',
    '52. Good afternoon': 'Good Afternoon',
    # Add more mappings as needed
    'Hello': 'Hello',
    'How are you': 'How are you',
    'Alright': 'Alright',
    'Good Morning': 'Good Morning',
    'Good Afternoon': 'Good Afternoon',
    'Good afternoon': 'Good Afternoon',
}

# All 16 words we want to train
ALL_WORDS = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

def organize_dataset(source_dir="Greetings", target_dir="training-data"):
    """
    Organize dataset from Greetings folder to training-data structure
    """
    print("="*70)
    print("📦 ORGANIZING DATASET FOR TRAINING")
    print("="*70)
    
    if not os.path.exists(source_dir):
        print(f"\n❌ Source directory '{source_dir}' not found!")
        return False
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Create folders for all 16 words
    for word in ALL_WORDS:
        word_dir = os.path.join(target_dir, word)
        os.makedirs(word_dir, exist_ok=True)
    
    print(f"\n📂 Source: {source_dir}")
    print(f"📂 Target: {target_dir}\n")
    
    # Process folders in source directory
    source_path = Path(source_dir)
    total_videos = 0
    organized_words = set()
    
    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        
        # Try to find mapping
        target_word = None
        if folder_name in FOLDER_MAPPING:
            target_word = FOLDER_MAPPING[folder_name]
        else:
            # Try to match by removing numbers/prefixes
            for key, value in FOLDER_MAPPING.items():
                if key.lower().replace('.', '').replace(' ', '') in folder_name.lower().replace('.', '').replace(' ', ''):
                    target_word = value
                    break
                # Check if folder name contains the word
                if value.lower() in folder_name.lower():
                    target_word = value
                    break
        
        if target_word is None:
            print(f"⚠️  Skipping '{folder_name}' - no mapping found")
            continue
        
        if target_word not in ALL_WORDS:
            print(f"⚠️  Skipping '{folder_name}' - '{target_word}' not in our 16 words")
            continue
        
        target_folder = os.path.join(target_dir, target_word)
        organized_words.add(target_word)
        
        # Copy videos
        video_count = 0
        video_files = list(folder.glob('*.MOV')) + list(folder.glob('*.mov')) + \
                     list(folder.glob('*.MP4')) + list(folder.glob('*.mp4')) + \
                     list(folder.glob('*.AVI')) + list(folder.glob('*.avi'))
        
        for video_file in video_files:
            try:
                target_file = os.path.join(target_folder, video_file.name)
                # Don't overwrite if exists
                if not os.path.exists(target_file):
                    shutil.copy2(video_file, target_file)
                    video_count += 1
                    total_videos += 1
            except Exception as e:
                print(f"    ❌ Error copying {video_file.name}: {e}")
        
        print(f"✅ {folder_name:30s} -> {target_word:15s} ({video_count} videos)")
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"✅ Total videos organized: {total_videos}")
    print(f"✅ Words with data: {len(organized_words)}")
    print(f"\n📁 Organized words:")
    for word in sorted(organized_words):
        word_dir = os.path.join(target_dir, word)
        count = len([f for f in os.listdir(word_dir) if f.lower().endswith(('.mov', '.mp4', '.avi'))])
        print(f"   • {word:15s}: {count:3d} videos")
    
    print("\n" + "="*70)
    print("✅ Dataset organization complete!")
    print("="*70)
    print(f"\n🚀 Ready to train! Run:")
    print(f"   python train-16words-model.py --data-dir {target_dir}")
    
    return True

if __name__ == '__main__':
    import sys
    
    source = sys.argv[1] if len(sys.argv) > 1 else "Greetings"
    target = sys.argv[2] if len(sys.argv) > 2 else "training-data"
    
    organize_dataset(source, target)

