# Dataset Structure for 16-Word Model Training

## Required Folder Structure

Place your video dataset in the following structure:

```
training-data/
├── Loud/
│   ├── video1.mp4
│   ├── video2.mp4
│   ├── video3.mp4
│   └── ... (more videos)
├── They/
│   ├── video1.mp4
│   └── ...
├── Sad/
│   └── ...
├── Quiet/
├── He/
├── Thank you/
├── How are you/
├── You/
├── It/
├── Good Afternoon/
├── Hello/
├── Alright/
├── Beautiful/
├── Happy/
├── None/
└── Good Morning/
```

## All 16 Word Folders Required:

1. **Loud**
2. **They**
3. **Sad**
4. **Quiet**
5. **He**
6. **Thank you**
7. **How are you**
8. **You**
9. **It**
10. **Good Afternoon**
11. **Hello**
12. **Alright**
13. **Beautiful**
14. **Happy**
15. **None**
16. **Good Morning**

## Video Requirements:

- **Format**: `.mp4`, `.avi`, `.mov` (any format OpenCV can read)
- **Duration**: 2-5 seconds recommended
- **Content**: Videos showing sign language gestures for each word
- **Minimum**: At least 10-20 videos per word (more is better!)

## How to Prepare Your Dataset:

1. Create the `training-data` folder in the project root
2. Create a folder for each of the 16 words
3. Place videos in the corresponding word folder
4. Make sure folder names match exactly (case-sensitive)

## Example:

```
training-data/
├── Hello/
│   ├── hello_001.mp4
│   ├── hello_002.mp4
│   └── hello_003.mp4
├── Thank you/
│   ├── thank_you_001.mp4
│   └── thank_you_002.mp4
└── ...
```

## Training Command:

Once your dataset is ready:

```bash
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"; python train-16words-model.py --data-dir training-data --epochs 100
```

## Tips:

- More videos per class = better model performance
- Try to have balanced dataset (similar number of videos per word)
- Videos should be clear and show full body/hands
- 2-5 second videos work best

