import cv2
import os
import shutil

def video2img(video_path):
    # Create output folder for frames
    output_folder = 'frames/' + '/'.join(video_path.split(os.path.sep)[-2:]).rsplit('.', 1)[0]
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
    # folder_path = 'src/batch_10_1/videos_batch_0.json/want'
    # if not os.path.exists(folder_path):
    #     print(f"Error: Path '{folder_path}' does not exist.")
    # else:
    #     for file in os.listdir(folder_path):
    #         if file.endswith('.mp4'):
    #             video_path = os.path.join(folder_path, file)
    #             video2img(video_path)

    main_folder = 'frames/want'
    for item in os.listdir(main_folder):
        subfolder = os.path.join(main_folder, item)
        if os.path.isdir(subfolder):
            if not os.listdir(subfolder):
                # Delete the subfolder after compression
                shutil.rmtree(subfolder)
                print(f"Deleted empty folder: {subfolder}")
            else:
                # Compress the subfolder
                archive_path = shutil.make_archive(subfolder, 'zip', subfolder)
                print(f"Compressed {subfolder} into {archive_path}")
                
                # Delete the subfolder after compression
                shutil.rmtree(subfolder)
                print(f"Deleted folder: {subfolder}")