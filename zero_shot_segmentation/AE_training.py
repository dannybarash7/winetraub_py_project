import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import numpy as np


# Dataset Definition
class OCTHistologyDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.data_files = self._load_data_files()

    def _load_data_files(self):
        data_mapping = {}
        for file in os.listdir(self.root_folder):
            if file.endswith(".npy"):
                key = "_".join(file.split("_")[2:])  # Extract unique identifier
                if key not in data_mapping:
                    data_mapping[key] = {}
                if "firstlayer_output" in file:
                    data_mapping[key]["target"] = os.path.join(self.root_folder, file)
                elif "dense_embeddings" in file:
                    data_mapping[key]["dense"] = os.path.join(self.root_folder, file)
                elif "image_pe" in file:
                    data_mapping[key]["image_pe"] = os.path.join(self.root_folder, file)
                elif "img_embed" in file:
                    data_mapping[key]["img_embed"] = os.path.join(self.root_folder, file)
                elif "sparse_embeddings" in file:
                    data_mapping[key]["sparse"] = os.path.join(self.root_folder, file)
        return [v for v in data_mapping.values() if "target" in v]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_entry = self.data_files[idx]
        dense = np.load(data_entry["dense"],allow_pickle=True)
        image_pe = np.load(data_entry["image_pe"],allow_pickle=True)
        img_embed = np.load(data_entry["img_embed"],allow_pickle=True)
        target = np.load(data_entry["target"],allow_pickle=True)

        input_data = np.concatenate([dense, image_pe, img_embed], axis=0)
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# 1. Define the Autoencoder (AE)
class SimpleAE(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleAE, self).__init__()

        # Compute flattened sizes
        self.input_dim = np.prod(input_shape)  # Flattened input
        self.output_dim = np.prod(output_shape)  # Flattened output
        self.output_shape = output_shape  # Store for reshaping later

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)  # Output layer size should match target
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten input
        x = x.view(batch_size, -1)  # Flatten from (B, C, H, W) to (B, input_dim)
        encoded = self.encoder(x)

        # Decode
        decoded = self.decoder(encoded)

        # Reshape output back to expected size
        decoded = decoded.view(batch_size, *self.output_shape)

        return decoded


# 2. Training Setup
root_folder = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct"
batch_size=1
if not os.path.exists(root_folder):
    #for cluster
    root_folder = "/home/barashd/Code/pytorch-CycleGAN-and-pix2pix/AE_training/data_of_oct"
    batch_size = 1
dataset = OCTHistologyDataset(root_folder=root_folder)
# Define training/validation split percentage
train_percent = 0.8
train_size = int(train_percent * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def filename_from_index(dataset,index):
    sample = dataset.data_files[index]
    sample_file = os.path.basename(sample["image_pe"])
    key = "_".join(sample_file.split("_")[2:])
    i = key.find("_jpg")
    return key[:i]

def print_filenames():
    global index, key
    print("Training files:")
    for index in train_dataset.indices:
        key = filename_from_index(dataset,index)
        print(f"{index}:{key}")
    print("Validation files:")
    for index in val_dataset.indices:
        key = filename_from_index(dataset,index)
        print(f"{index}:{key}")


print_filenames()
# Create DataLoaders
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define input and output dimensions dynamically
sample_input, sample_target = dataset[0]
input_dim = sample_input.shape[0]
output_dim = sample_target.shape[0]

# Initialize models and optimizer
ae = SimpleAE(input_shape=sample_input.shape, output_shape=sample_target.shape)
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=1e-4)

# Checkpoint directory
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint_path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)

def train_ae(num_epochs):
    ae.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for input_data, target in dataloader_train:
            optimizer.zero_grad()
            input_data, target = input_data.to(device), target.to(device)
            output = ae(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        # Validation step
        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for input_data, target in dataloader_val:
                input_data, target = input_data.to(device), target.to(device)
                output = ae(input_data)
                loss = criterion(output, target)
                val_loss += loss.item()
        ae.train()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(dataloader_train):.4f}, Validation Loss: {val_loss / len(dataloader_val):.4f}")

        # Save checkpoint
        save_checkpoint(epoch + 1, ae, optimizer, epoch_loss / len(dataloader_train))

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae.to(device)

# Train the AE
train_ae(num_epochs=80)
ae.eval()


def measure_performance():
    with torch.no_grad():
        for input_data, target in dataloader_val:
            input_data, target = input_data.to(device), target.to(device)
            ae_output = ae(input_data)
            filename = filename_from_index(dataset, index)
            path = f"{root_folder}/output/first_layer_output_{filename}"
            np.save(path, ae_output)

measure_performance()