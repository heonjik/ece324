import cv2
import os

def video2img(video_path):
    # Create output folder for frames
    output_folder = f"videos/{video_path}"
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Frame counter
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # End of video

        # Save frame as an image
        frame_path = os.path.join(output_folder, f'{frame_count:03d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")

if __name__=='__main__':
    video_path = 'video9.mp4'
    video2img(video_path)