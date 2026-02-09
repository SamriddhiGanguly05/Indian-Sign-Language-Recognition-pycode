"""Watch training progress in real-time"""
import os
import time
import subprocess
import sys

print("="*70)
print("LIVE TRAINING MONITOR")
print("="*70)
print("\nStarting training with live output...")
print("You will see:")
print("  - Dataset loading progress")
print("  - Each epoch progress")
print("  - Loss and accuracy metrics")
print("  - Best model saves")
print("\n" + "="*70 + "\n")

# Run training with live output
cmd = [
    sys.executable,
    "train-with-available-data.py",
    "--data-dir", "training-data",
    "--epochs", "100",
    "--batch-size", "8"
]

env = os.environ.copy()
env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

process = subprocess.Popen(
    cmd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

try:
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
except KeyboardInterrupt:
    print("\n\nStopping training...")
    process.terminate()
finally:
    process.wait()
    print("\n" + "="*70)
    print("Training process ended")
    print("="*70)

