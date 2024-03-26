import torch
import os
import random
from torchvision.transforms import Compose, Resize, ToTensor
from model import DualStream
from torch.utils.data import DataLoader
from main import MomentsInTimeDataset, read_video_frames
import numpy as np
# Assuming the same device configuration as main.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (Assuming the same transform as in main.py for consistency)
transform = Compose([
    Resize((64, 65)),  # Resize to the dimensions used in training
    ToTensor(),
])

def select_random_videos(data_dir, num_videos=1, same_category=False):
    """
    Selects random videos from the dataset. Can select from the same category if specified.
    """
    video_paths = []
    categories = os.listdir(data_dir)
    if same_category:
        selected_category = random.choice(categories)
        category_path = os.path.join(data_dir, selected_category)
        videos = os.listdir(category_path)
        videos = [os.path.join(category_path, video) for video in videos if video.endswith('.avi')]
        video_paths = random.sample(videos, min(num_videos, len(videos)))
    else:
        for _ in range(num_videos):
            selected_category = random.choice(categories)
            category_path = os.path.join(data_dir, selected_category)
            videos = os.listdir(category_path)
            videos = [os.path.join(category_path, video) for video in videos if video.endswith('.avi')]
            if videos:
                video_paths.append(random.choice(videos))
    return video_paths

def load_video(video_path):
    """
    Load a video and apply transformations.
    """
    # Assuming read_video_frames is implemented correctly
    frames = read_video_frames(video_path, transform, seq_len=5, num_seq=8, downsample=3)
    if frames is not None:
        frames = frames.unsqueeze(0)  # Add batch dimension
    return frames.to(device)

def attach_hooks(model):
    """
    Attach a hook to capture the output of the concat_hook_layer.
    """
    outputs = {}

    def hook(module, input, output):
        outputs['concat'] = output.detach()

    model.module.concat_hook_layer.register_forward_hook(hook)

    return outputs

def generate_output(model, video_paths, hook_outputs):
    """
    Process video through the model, capture the concatenated layer's output, and split it.
    """
    for video_path in video_paths:
        video_data = load_video(video_path)
        if video_data is None:
            print(f"Skipping video {video_path}, as it does not have enough frames.")
            continue

        # Forward pass
        with torch.no_grad():
            model(video_data)

        # Assuming the outputs are stored under 'concat_layer'
        concat_output = hook_outputs['concat_layer']

        # Split the concat_output tensor according to your model specifics.
        # This is a placeholder; you need to adjust the slicing according to your actual tensor shapes.
        stream1_output, stream2_output = torch.split(concat_output, split_size_or_sections=concat_output.size(1)//2, dim=1)

        print(f"Video: {video_path} - Stream1 Output Shape: {stream1_output.shape}, Stream2 Output Shape: {stream2_output.shape}")


def main():
    data_dir = '/home/zanh/ventral-dorsal-replication/UCF-101/UCF-101'  # Adjust to your video data directory

    # Initialize the model and load the trained weights
    model = DualStream().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    # Attach hooks to capture the outputs of the last layers of each stream
    hook_outputs = attach_hooks(model)

    # Example usage
    print("Generating outputs for random videos from different categories:")
    video_paths = select_random_videos(data_dir, num_videos=3, same_category=False)
    generate_output(model, video_paths, hook_outputs)

    print("\nGenerating outputs for random videos from the same category:")
    video_paths = select_random_videos(data_dir, num_videos=2, same_category=True)
    generate_output(model, video_paths, hook_outputs)

if __name__ == '__main__':
    main()
