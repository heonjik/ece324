import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def split_video_dataset(source_dir, train_dir, test_dir, train_ratio=0.8, seed=42):
    """
    Split a video dataset into training and testing sets while preserving class structure.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    class_videos = defaultdict(list)
    
    # Random seed for reproducibility
    random.seed(seed)
    
    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class directories
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        for root, _, files in os.walk(class_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    class_videos[class_name].append(os.path.join(root, file))
    
    # Tracking
    total_videos = 0
    total_train = 0
    total_test = 0
    
    # Split each class separately
    for class_name, videos in class_videos.items():
        # Shuffle
        random.shuffle(videos)
        
        # Split into training and testing sets
        split_idx = int(len(videos) * train_ratio)
        train_files = videos[:split_idx]
        test_files = videos[split_idx:]
        
        # Copy training files
        for file in train_files:
            filename = os.path.basename(file)
            shutil.copy2(file, os.path.join(train_dir, class_name, filename))
        
        # Copy testing files
        for file in test_files:
            filename = os.path.basename(file)
            shutil.copy2(file, os.path.join(test_dir, class_name, filename))
        
        # Update tracking
        total_videos += len(videos)
        total_train += len(train_files)
        total_test += len(test_files)
        
        print(f"Class '{class_name}': {len(videos)} videos → {len(train_files)} train, {len(test_files)} test")
    
    print(f"\nDataset split complete:")
    print(f"  - Total videos: {total_videos}")
    print(f"  - Training videos: {total_train} ({total_train/total_videos:.1%})")
    print(f"  - Testing videos: {total_test} ({total_test/total_videos:.1%})")
    print(f"  - Classes: {len(class_videos)}")

if __name__ == "__main__":
    ################# MODIFY ####################
    SOURCE_DIR = "src/batch_10_1/videos_batch_0.json"
    TRAIN_DIR = None
    TEST_DIR = None
    folder = "src/batch_10_1"
    files = ["videos_batch_0.json"]
    #############################################
    
    for classes in files:
        file_path = os.path.join(folder, classes)
        train_name = os.path.join(folder, f"train_{classes}")
        os.makedirs(train_name, exist_ok=True)
        TRAIN_DIR = train_name
        test_name = os.path.join(folder, f"test_{classes}")
        os.makedirs(test_name, exist_ok=True)
        TEST_DIR = test_name
    
    split_video_dataset(SOURCE_DIR, TRAIN_DIR, TEST_DIR)