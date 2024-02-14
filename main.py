import os
import torch
from torchvision.datasets.utils import download_url
from torchvision.datasets import UCF101
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from model import DualStream
import rarfile
import ssl
from PIL import Image
import cv2
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import random
import wandb
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = 'UCF-101'
ssl._create_default_https_context = ssl._create_unverified_context


criterion = nn.CrossEntropyLoss().to(device)

if not os.path.exists(data_dir):
    url = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
    download_url(url, root=data_dir, filename='UCF101.rar', md5=None)

    rar_path = 'UCF-101/UCF101.rar'

    try:
        os.makedirs(data_dir, exist_ok=True)
    except PermissionError:
        print("Permission denied: Failed to create the data directory.")

    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(data_dir)

transform = Compose([
    Resize((32, 32)),
    Grayscale(),
    ToTensor()
])

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, use_percentage=1.0, seq_len=10, num_seq=8, downsample=3):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.video_files = []
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if file.endswith('.avi'):
                        self.video_files.append(file_path)
        
        random.shuffle(self.video_files)
        num_files_to_use = int(len(self.video_files) * use_percentage)
        self.video_files = self.video_files[:num_files_to_use]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_frames = None
        attempts = 0
        while video_frames is None:
            video_path = self.video_files[(idx + attempts) % len(self.video_files)]
            video_frames = read_video_frames(video_path, self.transform, self.seq_len, self.num_seq, self.downsample)
            attempts += 1
            if attempts >= len(self.video_files):
                raise RuntimeError("Failed to find a video with enough frames")
        return {'video': video_frames}

def read_video_frames(video_path, transform, seq_len=10, num_seq=8, downsample=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = []

    if total_frames > 0:
        spacing = max(1, (total_frames - downsample * (seq_len - 1)) // num_seq)

    for seq_index in range(num_seq):
        start_frame = seq_index * spacing
        for frame_index in range(seq_len):
            if start_frame + frame_index * downsample < total_frames:
                frame_indices.append(start_frame + frame_index * downsample)
            else:
                # Not enough frames
                cap.release()
                return None  # Indicate insufficient frames

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = Image.fromarray(frame)
        if transform:
            frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames, dim=0).view(num_seq, seq_len, *frames[0].size())


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def main():
    wandb.init(project="Dual-Stream", config = {"learning_rate": 0.01, "epochs": 100, "batch_size": 4})
    
    dataset = VideoDataset(root_dir='UCF-101/UCF-101', transform=transform)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=16, drop_last=True)
    
    model = DualStream().to(device)
    inputs = next(iter(train_loader))['video'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6, amsgrad=True, eps=1e-8)

    num_epochs = 100
    log_interval = 20

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = batch['video'].to(device)

            score, mask = model(inputs)
            B = inputs.size(0)

            if i == 0: target, (_, B2, NS, NP, SQ) = process_output(mask)

            score_flattened = score.reshape(B*NP*SQ, B2*NS*SQ)
            target_flattened = target.reshape(B*NP*SQ, B2*NS*SQ)
            target_flattened_numerical = target_flattened.int()
            target_flattened = target_flattened_numerical.argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            
            running_loss += loss.item()

            if i % 1 == 0:
                print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')
                wandb.log({"train_loss": loss.item()})

            if i % 250 == 0:
                input_frame_to_log = inputs[0, :, 0, :, :].cpu()
                wandb.log({"example_input": [wandb.Image(input_frame_to_log, caption="Example Input Frame")]})

            optimizer.zero_grad()
            loss.backward()

            if(i % 500 == 0):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})

            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['video'].to(device)

                score, mask = model(inputs)
                B = inputs.size(0)
                
                score_flattened = score.reshape(B*NP*SQ, B2*NS*SQ)
                target_flattened = target.reshape(B*NP*SQ, B2*NS*SQ)
                target_flattened_numerical = target_flattened.int()
                target_flattened = target_flattened_numerical.argmax(dim=1)

                loss = criterion(score_flattened, target_flattened)
                val_loss += loss.item()
                wandb.log({"val_loss": loss / len(val_loader)})

        average_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1} Validation Loss: {average_val_loss}')

        checkpoint_path = 'model.pth'
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Saved model checkpoint at epoch {epoch + 1}")

    wandb.finish()

if __name__ == '__main__':
    main()
