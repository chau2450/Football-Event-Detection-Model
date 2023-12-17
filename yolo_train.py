
'''
This the training script ofr my YOLO V1 implementation

'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Football_Event_Detector.YOLO import yolo_v1, label, yolo_loss, train
from tqdm import tqdm
import os
import re


RESUME_TRAINING = True
LAST_SAVED_MODEL = 'YOLO_V1_train/yolo_epoch_278.pth.tar'  # Provide the path to your saved model



writer = SummaryWriter('YOLO_V1_train/runs/yolo_experiment_1')


# Hyperparameters
LEARNING_RATE = 2e-4  
DEVICE = "cuda" 
BATCH_SIZE = 64  # RTX 4090 should handle this; adjust if necessary
WEIGHT_DECAY = 1e-4
EPOCHS = 1000  # Adjust based on how quickly your model converges

# Initialize model, loss function, and optimizer
model = yolo_v1.Yolo().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = yolo_loss.Yolo_v1_loss()


def extract_epoch_from_filename(filename):
    # Extracts the epoch number from the filename using a regular expression
    match = re.search(r"epoch_(\d+).pth.tar", filename)
    return int(match.group(1)) if match else 0

# checks if some training was already done
start_epoch = 0
if RESUME_TRAINING:
    if os.path.isfile(LAST_SAVED_MODEL):
        print(f"Loading checkpoint '{LAST_SAVED_MODEL}'")
        checkpoint = torch.load(LAST_SAVED_MODEL)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Extract the epoch number from the filename
        start_epoch = extract_epoch_from_filename(LAST_SAVED_MODEL)
        print(f"Resumed training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at '{LAST_SAVED_MODEL}', starting training from scratch")


# Load Dataset
train_dataset = label.VOCDataset(
    csv_file='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/metadata/train.csv',
    img_dir='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/data/images',
    label_dir='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/data/labels', 
    S=7, B=2, C=20
)


train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True)


test_dataset = label.VOCDataset(
    csv_file='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/metadata/test.csv',
    img_dir='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/data/images',
    label_dir='/home/ivanbliminse/Documents/Football-Event-Detection-Model/src/Football_Event_Detector/YOLO/data/labels', 
    S=7, B=2, C=20
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

best_test_loss = float('inf')
best_model_info = {}

early_stopper = train.EarlyStopping(patience=20, verbose=True)

# save model function
def save_model(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def evaluate_model(model, test_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(test_loader)


for epoch in range(start_epoch,EPOCHS):
    model.train()  # Set the model to training mode
    batch_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

    for batch_idx, (images, targets) in batch_progress:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_progress.set_description(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

    # Validation step
    test_loss = evaluate_model(model, test_loader, loss_fn, DEVICE)
    writer.add_scalar('Test Loss', test_loss, epoch)

    # Call EarlyStopping
    early_stopper(test_loss, model, optimizer, epoch)

    # Check if early stopping was triggered
    if early_stopper.early_stop:
        print("Early stopping")
        break  # Break from the loop if early stopping is triggered

    # Refresh the progress bar for the next epoch
    batch_progress.close()



writer.close()
torch.cuda.empty_cache()