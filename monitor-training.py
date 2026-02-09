"""Monitor training progress"""
import os
import time
import subprocess

model_dir = "lstm-model-trained"

print("="*70)
print("TRAINING MONITOR - 200 EPOCHS")
print("="*70)

# Check if training is running
import psutil
python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                    if 'python' in p.info['name'].lower() and 
                    'train-with-available-data' in ' '.join(p.info.get('cmdline', []))]

if python_processes:
    print("\n✅ Training is RUNNING!")
    for p in python_processes:
        print(f"   Process ID: {p.info['pid']}")
else:
    print("\n⚠️  Training process not found")
    print("   It may have completed or not started yet")

# Check model files
print("\n" + "="*70)
print("MODEL FILES STATUS")
print("="*70)

if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    for f in files:
        filepath = os.path.join(model_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            mtime = time.ctime(os.path.getmtime(filepath))
            print(f"  {f:30s} {size:12d} bytes  {mtime}")
else:
    print("  Model directory does not exist yet")

print("\n" + "="*70)
print("Training will continue in background")
print("Check lstm-model-trained folder for updated files")
print("="*70)

