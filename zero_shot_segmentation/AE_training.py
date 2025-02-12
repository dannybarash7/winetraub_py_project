import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import numpy as np

# Set seeds for reproducibility
USE_AE = True
seed = 42  # Choose any fixed number
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using CUDA
random.seed(seed)
np.random.seed(seed)

checkpoint_path = "/cs/cs_groups/orenfr_group/barashd/train_ae/10.2/checkpoint_epoch_400.pth"
if not os.path.exists(checkpoint_path):
    checkpoint_path = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct/12.2/checkpoint_epoch_400.pth"
# checkpoint_path =""

# 2. Training Setup
root_folder = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct"
num_workers = 0
batch_size = 1
init_lr=1e-04
overwrite_checkpoint_lr = True
steps_between_applying_scheduler = 20
if not os.path.exists(root_folder):
    #for cluster
    root_folder = "/home/barashd/Code/pytorch-CycleGAN-and-pix2pix/AE_training/data_of_oct"

    batch_size = 20
    num_workers = 0
    print("Warning: setting number of workers to 0, for validation purposes")

training_files = []
validation_files = []
# with open("/home/barashd/Code/pytorch-CycleGAN-and-pix2pix/AE_training/Training_files.txt", "r") as f:
# with open("/Users/dannybarash/Code/oct/AE_experiment/Training_files.txt", "r") as f:
#     lines = f.readlines()
#     mode = None
#     for line in lines:
#         line = line.strip()
#         if line == "Training files:":
#             mode = "training"
#             continue
#         elif line == "Validation files:":
#             mode = "validation"
#             continue
#         elif line == "":
#             continue
#
#         if mode == "training":
#             training_files.append(line)
#         elif mode == "validation":
#             validation_files.append(line)



def load_checkpoint(model, optimizer, device, scheduler = None):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...",flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:  # Ensure scheduler state exists
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Checkpoint loaded successfully (Epoch: {epoch}, Loss: {loss:.4f})",flush=True)
        return epoch, loss
    else:
        print("No checkpoint found, starting from scratch.",flush=True)
        return 0, None  # Start from epoch 0 if no checkpoint is found


# Dataset Definition
class OCTHistologyDataset(Dataset):
    def __init__(self, root_folder, file_list=None):
        self.root_folder = root_folder
        self.data_files = self._load_data_files(file_list)

    def _load_data_files(self, file_list=None):
        data_mapping = {}
        for file in os.listdir(self.root_folder):
            if file.endswith(".npy"):
                key = "_".join(file.split("_")[4:])
                key = key.split("_jpg")[0]  # Remove additional suffix
                if file_list is not None and key not in file_list:
                    continue
                if key not in data_mapping:
                    data_mapping[key] = {}
                if "output_vhist" in file:  #imagenc_firstlayer_output_vhist
                    data_mapping[key]["target"] = os.path.join(self.root_folder, file)
                elif "input_oct" in file: #imagenc_firstlayer_input_oct
                    data_mapping[key]["input"] = os.path.join(self.root_folder, file)
        return [v for v in data_mapping.values() if "target" in v]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_entry = self.data_files[idx]
        input = np.load(data_entry["input"],allow_pickle=True)
        target = np.load(data_entry["target"],allow_pickle=True)

        input_data = np.concatenate(input, axis=0)
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# 1. Define the Autoencoder (AE)
class SimpleAE(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleAE, self).__init__()
        print("Using MLP:")
        # Compute flattened sizes
        self.input_dim = np.prod(input_shape)  # Flattened input
        self.output_dim = np.prod(output_shape)  # Flattened output
        self.output_shape = output_shape  # Store for reshaping later
        print(f"input_shape:{input_shape}, input_dim:{self.input_dim}, output_dim:{self.output_dim}, output_shape={output_shape}",flush=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Flatten input
        x = x.view(batch_size, -1)  # Flatten from (B, C, H, W) to (B, input_dim)
        decoded = self.mlp(x)

        # Decode
        # decoded = self.decoder(encoded)

        # Reshape output back to expected size
        decoded = decoded.view(batch_size, *self.output_shape)

        return decoded

class CNNVectorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels=64, kernel_size=3):
        super(CNNVectorTransformer, self).__init__()
        
        print("Using CNN:")
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               padding=1)

        # Fully connected layers to transform to output size
        self.fc1 = nn.Linear(input_dim * hidden_channels, 256)
        self.fc2 = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input to (batch, channels, length) for Conv1d
        x = x.unsqueeze(1)  # Shape: (batch, 1, input_dim)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten before feeding into fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Final output

        return x





dataset = OCTHistologyDataset(root_folder=root_folder)
# Define training/validation split percentage
train_percent = 0.8
train_size = int(train_percent * len(dataset))
val_size = len(dataset) - train_size
print(f"train_size:{train_size}",flush=True)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_dataset = OCTHistologyDataset(root_folder=root_folder, file_list=training_files)
# val_dataset = OCTHistologyDataset(root_folder=root_folder, file_list=validation_files)

def filename_from_index(dataset,index):
    sample = dataset.data_files[index]
    sample_file = os.path.basename(sample["target"])
    key = "_".join(sample_file.split("_")[4:])
    i = key.find("_jpg")
    return key[:i]

def print_filenames():
    global index, key
    print(f"Training files:{training_files}",flush=True)
    for index in train_dataset.indices:
        key = filename_from_index(dataset,index)
        print(f"{index}:{key}",flush=True)
    print(f"Validation files:{validation_files}",flush=True)

    for index in val_dataset.indices:
        key = filename_from_index(dataset,index)
        print(f"{index}:{key}",flush=True)


print_filenames()
# Create DataLoaders

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define input and output dimensions dynamically
sample_input, sample_target = train_dataset[0]
# input_dim = sample_input.shape[0]
# output_dim = sample_target.shape[0]

# Initialize models and optimizer
# Load model and optimizer
model = None
if USE_AE:
    model = SimpleAE(input_shape=sample_input.shape, output_shape=sample_target.shape)
else:
    # Example usage
    input_dim = np.prod(sample_input.shape)  # Flattened input
    output_dim = np.prod(sample_target.shape)  # Flattened output
    model = CNNVectorTransformer(input_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# Define scheduler before loading checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint if available
#scheduler=None for cpu/cuda match
print("loading checkpoint",flush=True)
start_epoch, _ = load_checkpoint(model, optimizer, device, scheduler=None)
# Override the learning rate after loading the checkpoint
if overwrite_checkpoint_lr:
    new_lr = init_lr  # Set your desired learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


criterion = nn.MSELoss()
# Move model to the appropriate device
print("Move to device",flush=True)
model.to(device)
print("define LR...",flush=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # Because you want to reduce when error doesn't improve
    factor=0.5,  # Reduce LR by multiplying with this factor
    patience=10,  # Wait for 15 epochs before reducing LR
    threshold=1e-4,  # Minimum improvement required to reset patience
    threshold_mode='rel',  # Relative to previous best
    verbose=True  # Print LR updates
)# Move optimizer state tensors to the correct device after loading state dict

for param_group in optimizer.param_groups:
    param_group['params'] = [p.to(device) for p in param_group['params']]

for state in optimizer.state.values():
    if isinstance(state, torch.Tensor):
        state.data = state.data.to(device)
    elif isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
# Checkpoint directory
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(epoch, model, optimizer, scheduler, loss):
    checkpoint_path = os.path.join(checkpoint_dir, f"/cs/cs_groups/orenfr_group/barashd/train_ae/10.2/checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
        'loss': loss
    }, checkpoint_path)

def train_ae(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for step,(input_data, target) in enumerate(dataloader_train):
            optimizer.zero_grad()
            input_data, target = input_data.to(device), target.to(device)
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_data, target in dataloader_val:
                input_data, target = input_data.to(device), target.to(device)
                output = model(input_data)
                loss = criterion(output, target)
                val_loss += loss.item()
        model.train()
        scheduler.step(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(dataloader_train)}, Validation Loss: {val_loss / len(dataloader_val)}, LR: {scheduler.get_last_lr()[0]}",flush=True)

        # Save checkpoint
        if (epoch +1) % 100 == 0:
            save_checkpoint(epoch + 1, model, optimizer, scheduler, epoch_loss / len(dataloader_train))

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",flush=True)
model.to(device)
print("Moved to device",flush=True)
# Train the AE
try:
    train_ae(num_epochs=1000)
    pass
except Exception as e:
    print(f"Error occurred: {e}", flush=True)
model.eval()
def measure_performance():

    root_path = "/cs/cs_groups/orenfr_group/barashd/train_ae/12.2/ae_output_validation"
    if not os.path.exists("/cs/cs_groups/orenfr_group/barashd/"):
        root_path = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct/12.2/ae_output_validation"

    os.makedirs(root_path , exist_ok=True)
    print(f"root_path:{root_path}",flush=True)
    with torch.no_grad():
        for index, (input_data, target) in enumerate(dataloader_val):
            input_data, target = input_data.to(device), target.to(device)
            ae_output = model(input_data).detach().cpu().numpy()
            # filename = validation_files[index]
            filename = filename_from_index(dataset, index)
            path = f"{root_path}/predicted_firstlayer_output_{filename}"
            print(f"saved {path}",flush=True)
            np.save(path, ae_output)
        for index, (input_data, target) in enumerate(dataloader_train):
            input_data, target = input_data.to(device), target.to(device)
            ae_output = model(input_data).detach().cpu().numpy()
            # filename = validation_files[index]
            filename = filename_from_index(dataset, index)
            path = f"{root_path}/predicted_firstlayer_output_{filename}"
            print(f"saved {path}",flush=True)
            np.save(path, ae_output)
try:
#    measure_performance()
    pass
except Exception as e:
    print(f"Error occurred: {e}", flush=True)
