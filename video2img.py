import cv2
import os

def video2img(video_path, output_base, class_label):
    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create output folder for frames: testing_videos_non_masked/<class_label>/<video_name>/
    output_folder = os.path.join(output_base, class_label, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Frame counter
    frame_count = 0
    frame_skip = 5
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_path = os.path.join(output_folder, f'{frame_count:03d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from '{video_name}' to '{output_folder}'")

if __name__ == '__main__':
    # Base folder containing class folders (e.g., 'teacher/')
    folder_path = 'src/batch_10_1/test_videos_test_batch_0.json'
    output_base = 'testing_videos_non_masked'
    class_label = 'happy'

    class_folder = os.path.join(folder_path, class_label)

    for file_name in os.listdir(class_folder):
        if file_name.lower().endswith('.mp4'):
            video_file = os.path.join(class_folder, file_name)
            video2img(video_file, output_base, class_label)
            

    
    