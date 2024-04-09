import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from model import DualStream
from single_stream_model import SingleStream
from main import read_video_frames
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# Set the device to the second GPU
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define the transform for the video frames
transform = Compose([
    Resize((64, 65)),
    ToTensor(),
])

# Wrapper class for the DualStream model
class DualStreamWrapper(nn.Module):
    def __init__(self, dual_stream_model):
        super(DualStreamWrapper, self).__init__()
        self.dual_stream = dual_stream_model
    
    def forward(self, x):
        return self.dual_stream(x)

# Function to load the model
def load_model():
    model_path = 'models/model_epoch_73.pth'
    dual_stream_model = DualStream().to(device)
    state_dict = torch.load(model_path, map_location=device)
    dual_stream_model.load_state_dict(state_dict, strict=False)
    dual_stream_model.eval()
    return DualStreamWrapper(dual_stream_model)

# Function to select videos from each category
def select_videos_from_each_category(data_dir, num_videos_per_category=30, max_categories=7):
    video_paths = {}
    categories = os.listdir(data_dir)
    selected_categories = random.sample(categories, min(len(categories), max_categories))

    for category in selected_categories:
        print(category)
        category_path = os.path.join(data_dir, category)
        videos = os.listdir(category_path)
        videos = [os.path.join(category_path, video) for video in videos if video.endswith('.avi')]
        if len(videos) >= num_videos_per_category:
            video_paths[category] = random.sample(videos, num_videos_per_category)
        else:
            video_paths[category] = videos

    return video_paths

# Function to load a video and apply transformations
def load_video(video_path):
    frames = read_video_frames(video_path, transform, seq_len=5, num_seq=8, downsample=3)
    if frames is not None:
        frames = frames.unsqueeze(0)
    return frames.to(device)

# Function to generate latent spaces and labels
def generate_latent_space_and_labels(model, video_paths, category_label):
    latent_space = []
    latent_space_part1 = []
    latent_space_part2 = []
    labels = []
    for video_path in video_paths:
        video_data = load_video(video_path)
        print(video_path)
        if video_data is None:
            print(f"Skipping video {video_path}, as it does not have enough frames.")
            continue

        with torch.no_grad():
            _, _, concat_output, future_context = model(video_data)
            #comment out the below line for dual stream model
        
        # Split the context vector into two parts
        part1, part2 = torch.split(future_context, concat_output.size(1)//2, dim=1)
        latent_space.append(future_context.cpu().numpy().flatten())
        latent_space_part1.append(part1.cpu().numpy().flatten())
        latent_space_part2.append(part2.cpu().numpy().flatten())
        labels.append(category_label)
    return np.array(latent_space), np.array(latent_space_part1), np.array(latent_space_part2), labels

# Function to plot t-SNE
def plot_tsne(latent_spaces, labels, title, filename):
    print(f"Computing t-SNE for {title}")
    tsne = TSNE(n_components=2, perplexity=2, random_state=42, n_iter=5000)
    reduced_data = tsne.fit_transform(latent_spaces)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate a color map
    num_unique_labels = len(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, num_unique_labels))
    label_to_color = {label: color for label, color in zip(set(labels), colors)}

    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], color=label_to_color[label], label=label)

    # Add color bar
    color_map = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(0, num_unique_labels))
    cbar = fig.colorbar(color_map, ax=ax, boundaries=np.arange(-0.5, num_unique_labels), ticks=np.arange(0, num_unique_labels))
    cbar.set_ticklabels(list(set(labels)))

    ax.set_title(title)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(loc='best')

    plt.savefig(f'{filename}.png')
    plt.close(fig)

# Main function to run the model and generate t-SNE plots
def main():
    data_dir = '/home/zanh/ventral-dorsal-replication/UCF-101/UCF-101'
    wrapped_model = load_model()

    categories_video_paths = select_videos_from_each_category(data_dir)
    all_latent_spaces = []
    all_latent_spaces_part1 = []
    all_latent_spaces_part2 = []
    all_labels = []

    for category, video_paths in categories_video_paths.items():
        latent_space, latent_space_part1, latent_space_part2, labels = generate_latent_space_and_labels(wrapped_model, video_paths, category)
        all_latent_spaces.extend(latent_space)
        all_latent_spaces_part1.extend(latent_space_part1)
        all_latent_spaces_part2.extend(latent_space_part2)
        all_labels.extend(labels)

    # Plot the combined context vector
    plot_tsne(np.array(all_latent_spaces), all_labels, 'Combined Context Vector', 'tsne_combined_context')
    
    # Plot the first part of the context vector
    plot_tsne(np.array(all_latent_spaces_part1), all_labels, 'First Part of Context Vector', 'tsne_context_part1')

    # Plot the second part of the context vector
    plot_tsne(np.array(all_latent_spaces_part2), all_labels, 'Second Part of Context Vector', 'tsne_context_part2')

if __name__ == '__main__':
    main()
