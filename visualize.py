"""import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from model import DualStream
from main import read_video_frames
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define the transformation for the video frames
transform = Compose([
    Resize((64, 65)),
    ToTensor(),
])

def load_model(model_path='models/model_epoch_6.pth'):
    model = DualStream().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_video(video_path):
    frames = read_video_frames(video_path, transform, seq_len=5, num_seq=8, downsample=3)
    if frames is not None:
        frames = frames.unsqueeze(0)
    return frames.to(device)

def get_future_context(model, video_path):
    video_data = load_video(video_path)
    if video_data is None:
        return None
    with torch.no_grad():
        _, _, _, future_context = model(video_data)
    return future_context.squeeze().cpu().numpy()

def main():
    data_dir = '/home/zanh/ventral-dorsal-replication/UCF-101/UCF-101'
    model = load_model()

    # Create a dictionary to hold one video per category
    category_videos = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.avi'):
                category = os.path.basename(root)
                if category not in category_videos:
                    category_videos[category] = os.path.join(root, file)

    # Randomly select one video and prepare the sample space
    selected_video = random.choice(list(category_videos.values()))
    sample_space = [v for k, v in category_videos.items() if v != selected_video]

    ref_context = get_future_context(model, selected_video)
    other_contexts = np.array([get_future_context(model, vp) for vp in sample_space])

    # Compute cosine similarity
    similarity = cosine_similarity([ref_context], other_contexts)[0]
    closest_indices = np.argsort(similarity)[-3:]  # Get indices of 3 highest values

    print(f"Reference video: {selected_video}")
    print("Closest videos:")
    for idx in closest_indices:
        print(sample_space[idx])

if __name__ == '__main__':
    main()"""

import os
import subprocess
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from model import DualStream  # Ensure this is the model you want to use
from main import read_video_frames  # Ensure this function exists and works as expected
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity

# Set device for model computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path='models/model_epoch_73.pth'):
    model = DualStream().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_video(video_path, transform):
    frames = read_video_frames(video_path, transform, seq_len=5, num_seq=8, downsample=3)
    if frames is not None:
        frames = frames.unsqueeze(0).to(device)
    return frames

def get_future_context(model, video_path, transform):
    video_data = load_video(video_path, transform)
    if video_data is None:
        return None
    with torch.no_grad():
        _, _, _, future_context = model(video_data)
    return future_context.cpu().numpy()

def convert_video_to_gif(video_path, start_time=0, duration=5, gif_path='output.gif'):
    """
    Convert a video file to a GIF using ffmpeg.
    The start_time and duration specify the video segment to convert.
    """
    command = [
        'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration), '-i', video_path,
        '-vf', 'fps=10,scale=320:-1', '-gifflags', '+transdiff', gif_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def combine_gifs(gif_paths, output_path='combined.gif', text_list=None):
    """
    Combine several GIFs into one and add text annotations.
    """
    images = [Image.open(gif) for gif in gif_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for im in images:
        combined_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Add text annotations if provided
    if text_list:
        draw = ImageDraw.Draw(combined_image)
        font = ImageFont.load_default()
        y_text = max_height - 20
        for i, text in enumerate(text_list):
            draw.text((sum(widths[:i]) + 5, y_text), text, font=font, fill=(255,255,255,255))

    combined_image.save(output_path)

def main(runs=5, num_samples=500):
    data_dir = '/home/zanh/ventral-dorsal-replication/UCF-101/UCF-101'
    model = load_model()
    transform = Compose([Resize((64, 65)), ToTensor()])

    for run in range(runs):
        video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if f.endswith('.avi')]
        selected_video = random.choice(video_files)
        sample_videos = random.sample(video_files, num_samples)

        ref_context = get_future_context(model, selected_video, transform)
        if ref_context is not None:
            ref_context = ref_context.flatten()

        contexts = []
        for video in sample_videos:
            context = get_future_context(model, video, transform)
            if context is not None:
                contexts.append(context.flatten())

        if contexts:
            # Compute cosine similarity
            similarities = cosine_similarity([ref_context], contexts)[0]
            top_indices = np.argsort(similarities)[-3:]

            top_videos = [sample_videos[i] for i in top_indices]
            print(f"Selected video: {selected_video}")
            print("Closest videos:", top_videos)

            # Convert to GIFs
            gif_paths = []
            for i, video in enumerate([selected_video] + top_videos):
                gif_path = f'run{run}_video{i}.gif'
                convert_video_to_gif(video, gif_path=gif_path)
                gif_paths.append(gif_path)

            # Combine GIFs
            combine_gifs(gif_paths, output_path=f'combined_run{run}.gif', text_list=[os.path.basename(v) for v in [selected_video] + top_videos])


if __name__ == '__main__':
    main()
