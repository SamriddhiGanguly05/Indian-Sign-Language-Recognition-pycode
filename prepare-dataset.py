"""
Helper script to check and prepare your dataset structure
Run this before training to verify your dataset is ready
"""

import os

WORDS = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

def check_dataset(data_dir="training-data"):
    """Check if dataset structure is correct"""
    print("="*70)
    print("🔍 CHECKING DATASET STRUCTURE")
    print("="*70)
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Directory '{data_dir}' does not exist!")
        print(f"\n📋 Please create the folder structure:")
        print(f"   {data_dir}/")
        for word in WORDS:
            print(f"   ├── {word}/")
        return False
    
    print(f"\n✅ Found directory: {data_dir}\n")
    
    all_good = True
    total_videos = 0
    
    for word in WORDS:
        word_dir = os.path.join(data_dir, word)
        
        if not os.path.exists(word_dir):
            print(f"❌ Missing folder: {word}")
            all_good = False
            continue
        
        video_files = [f for f in os.listdir(word_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if len(video_files) == 0:
            print(f"⚠️  {word:15s}: 0 videos (EMPTY!)")
            all_good = False
        else:
            print(f"✅ {word:15s}: {len(video_files):3d} videos")
            total_videos += len(video_files)
    
    print("\n" + "="*70)
    print(f"📊 Total Videos Found: {total_videos}")
    print("="*70)
    
    if all_good and total_videos > 0:
        print("\n✅ Dataset structure is correct! Ready for training.")
        print(f"\n🚀 Run training with:")
        print(f"   python train-16words-model.py --data-dir {data_dir}")
        return True
    else:
        print("\n⚠️  Dataset needs attention:")
        if total_videos == 0:
            print("   - No videos found in any folder")
        print("   - Make sure all 16 word folders exist")
        print("   - Add videos to each folder")
        return False

if __name__ == '__main__':
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "training-data"
    check_dataset(data_dir)

