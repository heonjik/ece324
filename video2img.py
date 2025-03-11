import cv2
import os

def video2img(video_path):
    # Create output folder for frames
    output_folder = f"training_videos_non_masked/want/{video_path}"
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Frame counter
    frame_count = 0
    frame_skip = 5

    while cap.isOpened():
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # End of video
        
        
        if frame_count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f'{frame_count:03d}.jpg')
            cv2.imwrite(frame_path, frame)
        frame_count += 1
        

    cap.release()
    print(f"Extracted {frame_count/frame_skip} frames to '{output_folder}'")


if __name__ == '__main__':
    # Folder containing your mp4 files
    folder_path = 'src/batch_10_1/videos_batch_0.json/want'
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.mp4'):
            video_file = os.path.join(folder_path, file_name)
            video2img(video_file)
            

    
    