import pandas as pd
import json
import os
import requests
import subprocess
from pytube import YouTube

def clip_download_video(url, start_time, end_time, output_path):
    temp_path = "temp_video.mp4"
    try:
        print(f"Downloading video from {url}...")
        yt = YouTube('https://www.youtube.com/watch?v=mfrGIHTnKvI') # supposed to be url but manually using valid url for testing
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream.download(filename=temp_path)
        print("Download complete")
    except Exception as e:
        print(f"Error downloading: {str(e)}")
        return False
    
    try:
        print(f"Clipping video from {start_time}s to {end_time}s...")
        command = [
            'ffmpeg',
            '-i', temp_path,           # Input file
            '-ss', str(start_time),    # Start time
            '-to', str(end_time),      # End time
            '-c:v', 'libx264',         # Video codec
            '-c:a', 'aac',             # Audio codec
            '-strict', 'experimental', # For older ffmpeg versions
            output_path                # Output file
        ]
        subprocess.run(command, check=True)
        print(f"Successfully created clip at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during clipping: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return True

def collect_videos(batch_data, batch_classes, output_path):
    video_count = 0
    for class_name in batch_classes:
        class_folder = os.path.join(output_path, class_name)
        os.makedirs(class_folder, exist_ok=True)

    for data in batch_data:
        text = data['text']
        start_t = data['start_time']
        end_t = data['end_time']
        url = data['url']
        
        video_filename = f"video_{video_count}.mp4"
        video_path = os.path.join(class_folder, video_filename)
        
        clip_download_video(url, start_t, end_t, video_path)
        video_count += 1
        

if __name__=="__main__":
    ############################
    folder = "src/batch_10_1"
    ############################
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        folder_name = os.path.join(folder, f"videos_{file}")
        os.makedirs(folder_name, exist_ok=True)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            if "data" in data and "class_names" in data:
                batch_data = data["data"]
                batch_classes = data["class_names"]
            else:
                print(f"key not found in {file}")
        collect_videos(batch_data, batch_classes, folder_name)