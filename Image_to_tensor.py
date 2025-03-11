import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def _is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

import torch
from torch.nn.utils.rnn import pad_sequence

def my_collate_fn(batch):
    # Each element in the batch is a tuple: (video_tensor, label)
    videos, labels = zip(*batch)
    
    # videos is a list of tensors with shapes like (T, C, H, W) where T can differ.
    # pad_sequence will pad along the first dimension (time dimension)
    padded_videos = pad_sequence(videos, batch_first=True)  # Shape: (B, T_max, C, H, W)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels)
    return padded_videos, labels


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all videos, structured as:
                            root_dir/class_name/video_id/*.jpg (or .png, etc.)
            transform (callable, optional): Transform to apply to each frame.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_dirs = []  # List of paths to each video folder
        self.labels = []      # Corresponding label for each video
        self.label2idx = {}   # Mapping class name -> numeric label
        
        # Loop through each class folder
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                # Assign a numeric label for the class if needed
                if class_name not in self.label2idx:
                    self.label2idx[class_name] = len(self.label2idx)
                # Loop through each video directory
                for video_id in os.listdir(class_path):
                    video_path = os.path.join(class_path, video_id)
                    if os.path.isdir(video_path):
                        # List all allowed image files
                        frame_files = [f for f in os.listdir(video_path) if _is_image_file(f)]
                        if len(frame_files) == 0:
                            #print(f"Skipping empty video directory: {video_path}")
                            continue
                        self.video_dirs.append(video_path)
                        self.labels.append(self.label2idx[class_name])
                        
    def __len__(self):
        return len(self.video_dirs)
    
    def __getitem__(self, idx):
        video_path = self.video_dirs[idx]
        label = self.labels[idx]
        
        # Get a sorted list of image files
        frame_files = sorted([f for f in os.listdir(video_path) if _is_image_file(f)])
        if not frame_files:
            # This should not happen if __init__ filtering works, but just in case
            raise ValueError(f"No image files found in {video_path}")
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            img = Image.open(frame_path).convert('L')  # Convert to grayscale
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        # Stack frames along the time dimension: (T, C, H, W)
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, label

# Define image transforms (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.CenterCrop((224,400)),
    transforms.ToTensor()
])

# Use the correct root folder (adjust the name to match your folder)
root_folder = 'training_videos_non_masked'
dataset = VideoDataset(root_dir=root_folder, transform=transform)

# If after initialization the dataset is empty, double-check your folder structure.
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check that your root directory exists and contains non-empty video folders.")

dataloader = DataLoader(dataset, batch_size=15, shuffle=True, collate_fn=my_collate_fn)

# Iterate over the dataloader
for video_tensor, label in dataloader:
    print("Video tensor shape:", video_tensor.shape)  # Expected: (batch_size, T, C, H, W)
    print("Labels:", label)
    num_frames = 12
    to_pil = ToPILImage()

    num_rows = (num_frames + 2) // 3  # Round up to ensure all frames fit

    # Create a figure with subplots: num_rows rows, 3 columns
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i in range(num_frames):
        single_frame = video_tensor[0, i, :, :, :]  # Get the i-th frame of the first video
        pil_image = to_pil(single_frame)
        axes[i].imshow(pil_image, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')

    # Hide any unused subplots
    for j in range(num_frames, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
    break
    


