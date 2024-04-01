import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from model import DualStream
from main import read_video_frames
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
import matplotlib.cm as cm

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

transform = Compose([
    Resize((64, 65)),
    ToTensor(),
])

class DualStreamWrapper(nn.Module):
    def __init__(self, dual_stream_model):
        super(DualStreamWrapper, self).__init__()
        self.dual_stream = dual_stream_model
    
    def forward(self, x):
        return self.dual_stream(x)

def load_model():
    model_path = 'model.pth'
    dual_stream_model = DualStream().to(device)
    state_dict = torch.load(model_path, map_location=device)
    dual_stream_model.load_state_dict(state_dict, strict=False)
    dual_stream_model.eval()
    return DualStreamWrapper(dual_stream_model)

def select_videos_from_each_category(data_dir, num_videos_per_category=2):
    video_paths = {}
    categories = os.listdir(data_dir)
    for category in categories:
        category_path = os.path.join(data_dir, category)
        videos = os.listdir(category_path)
        videos = [os.path.join(category_path, video) for video in videos if video.endswith('.avi')]
        if len(videos) >= num_videos_per_category:
            video_paths[category] = random.sample(videos, num_videos_per_category)
        else:
            video_paths[category] = videos
    return video_paths

def load_video(video_path):
    frames = read_video_frames(video_path, transform, seq_len=5, num_seq=8, downsample=3)
    if frames is not None:
        frames = frames.unsqueeze(0)
    return frames.to(device)

def generate_latent_space(model, video_paths):
    latent_space = []
    for video_path in video_paths:
        video_data = load_video(video_path)
        if video_data is None:
            print(f"Skipping video {video_path}, as it does not have enough frames.")
            continue

        with torch.no_grad():
            _, _, concat_output = model(video_data)
        latent_space.append(concat_output.cpu().numpy().flatten())
    return np.array(latent_space)

def generate_latent_space_and_labels(model, video_paths, category_label):
    latent_space = []
    labels = []
    for video_path in video_paths:
        video_data = load_video(video_path)
        if video_data is None:
            print(f"Skipping video {video_path}, as it does not have enough frames.")
            continue

        with torch.no_grad():
            _, _, concat_output = model(video_data)
        latent_space.append(concat_output.cpu().numpy().flatten())
        labels.append(category_label)
    return np.array(latent_space), labels

def plot_tsne(latent_spaces, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(latent_spaces)
    
    # Create a subplot for the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate a color map based on the number of unique labels
    num_unique_labels = len(set(labels))
    colors = cm.rainbow(np.linspace(0, 1, num_unique_labels))
    label_to_color = dict(zip(set(labels), colors))
    
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], color=label_to_color[label], label=label)
    
    # Create a color bar with the label-to-color mapping
    color_map = plt.cm.ScalarMappable(cmap=cm.rainbow)
    color_map.set_array([])
    fig.colorbar(color_map, ticks=np.linspace(0, 1, num_unique_labels), boundaries=np.arange(0, 1.1, 1/num_unique_labels))

    ax.set_title('t-SNE of Video Latent Spaces')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.legend(loc='best')
    
    plt.savefig('tsne_distribution.png')

def main():
    data_dir = '/home/zanh/ventral-dorsal-replication/UCF-101/UCF-101'
    wrapped_model = load_model()

    categories_video_paths = select_videos_from_each_category(data_dir)
    all_latent_spaces = []
    all_labels = []

    for category, video_paths in categories_video_paths.items():
        latent_spaces, labels = generate_latent_space_and_labels(wrapped_model, video_paths, category)
        all_latent_spaces.extend(latent_spaces)
        all_labels.extend(labels)

    plot_tsne(np.array(all_latent_spaces), all_labels)

if __name__ == '__main__':
    main()
