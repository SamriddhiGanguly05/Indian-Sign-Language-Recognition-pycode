"""Check training status"""
import os
import time

model_dir = "lstm-model-trained"

print("="*70)
print("TRAINING STATUS CHECK")
print("="*70)

if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    if files:
        print(f"\nModel directory exists: {model_dir}")
        print("\nFiles found:")
        for f in files:
            filepath = os.path.join(model_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                mtime = time.ctime(os.path.getmtime(filepath))
                print(f"  - {f:30s} Size: {size:10d} bytes  Modified: {mtime}")
        
        if "best_model.hdf5" in files or "final_model.hdf5" in files:
            print("\n✅ Training appears to be complete or in progress!")
        else:
            print("\n⏳ Training may still be running...")
    else:
        print(f"\n⚠️  Model directory exists but is empty")
        print("   Training may be starting...")
else:
    print(f"\n⚠️  Model directory '{model_dir}' does not exist yet")
    print("   Training may not have started or is just beginning")

print("\n" + "="*70)
print("To see training in real-time, run in a new terminal:")
print("="*70)
print("  $env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python';")
print("  python train-with-available-data.py --data-dir training-data --epochs 100 --batch-size 8")
print("="*70)

