from PIL import Image
import torch
import numpy as np
import clip
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Python script to fine-tune CLIP for pictorial locations
# Adopted and adapted from: https://github.com/statscol/clip-fine-tuning/blob/main/clip-fine-tuning.ipynb

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

area_captions = np.load("pictorialMapsCaptionsLocation.npy")
area_paths = np.load("pictorialMapsPathsLocation.npy")

print(Counter(area_captions))

# Define train, validation and test splits

n_splits = 1
test_size = 0.1
val_size = 0.1

captions = area_captions
image_paths = area_paths

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

for train_index, test_index in sss.split(image_paths, captions):

    train_captions, train_image_paths = [captions[i] for i in train_index], [
        image_paths[i] for i in train_index
    ]
    test_captions, test_image_paths = [captions[i] for i in test_index], [
        image_paths[i] for i in test_index
    ]


# second split for validation set

sss_val = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=42)

for train_index, val_index in sss_val.split(train_image_paths, captions[train_index]):

    train_captions_final, train_image_paths_final = [
        train_captions[i] for i in train_index
    ], [train_image_paths[i] for i in train_index]
    val_captions, val_image_paths = [train_captions[i] for i in val_index], [
        train_image_paths[i] for i in val_index
    ]


print(len(train_captions_final))
print(len(val_captions))
print(len(test_captions))

print(Counter(train_captions_final))
print(Counter(val_captions))
print(Counter(test_captions))


class Countries(Dataset):
    def __init__(self, captions, list_image_path):
        self.captions = captions
        self.image_path = list_image_path

    def __getitem__(self, idx):

        image = preprocess(Image.open(self.image_path[idx]))
        caption = self.captions[idx]
        return {"image": image, "caption": caption}

    def __len__(self):
        return len(self.captions)


train_dataset = Countries(train_captions_final, train_image_paths_final)
val_dataset = Countries(val_captions, val_image_paths)


print(len(train_dataset))


BATCH_SIZE = 10
tr_dl = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
ts_dl = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)


N_EPOCHS = 32
loss_img = nn.CrossEntropyLoss()
loss_caption = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(tr_dl) * N_EPOCHS)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


print(DEVICE)
model.to(DEVICE)


# to avoid problems with mixed precision, taken from here https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# Fine-tune CLIP. Only save best model

def train_model(
    n_epochs, train_dataloader, test_dataloader, checkpoint_path: str = "./checkpoints"
):
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    print(f"Using {DEVICE} for training")
    best_score = 9999999
    history = {"train_loss": [], "val_loss": []}
    for epoch in tqdm(range(n_epochs)):
        total_steps = 0
        train_loss = 0.0
        model.train()
        for step, data in enumerate(train_dataloader, 1):

            optimizer.zero_grad()

            img_batch = data["image"].to(DEVICE)
            captions_batch = clip.tokenize(data["caption"], truncate=True).to(DEVICE)
            with torch.cuda.amp.autocast():
                logits_image, logits_caption = model(img_batch, captions_batch)
            labels = torch.arange(len(img_batch)).to(
                DEVICE
            )  # we are interested on predicting the right caption which is the caption position of every image
            img_loss = loss_img(logits_image, labels)
            caption_loss = loss_caption(logits_caption, labels)
            total_loss = (img_loss + caption_loss) / 2
            total_loss.backward()
            train_loss += total_loss.item()
            convert_models_to_fp32(model)
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_steps += 1
            scheduler.step()  # per step scheduler
            if step % 100 == 0:
                print(f"Epoch {epoch} step loss:{train_loss / total_steps}")
                print(f"Lr at step {step}: {optimizer.param_groups[0]['lr']:.5f}")

        history["train_loss"].append(train_loss / len(train_dataloader))
        val_metrics = validate(test_dataloader)
        history["val_loss"].append(val_metrics)
        if val_metrics < best_score:
            print("Better score reached, saving checkpoint...")
            if os.path.exists(Path(checkpoint_path) / "best_model.pt"):
                os.remove(Path(checkpoint_path) / "best_model.pt")
            best_score = val_metrics
            torch.save(model.state_dict(), Path(checkpoint_path) / "best_model.pt")

    return history


def validate(test_dl):
    model.eval()
    test_loss = 0.0
    for data in tqdm(test_dl, desc="Evaluating in validation set"):
        img_batch = data["image"].to(DEVICE)
        captions_batch = clip.tokenize(data["caption"], truncate=True).to(DEVICE)
        with torch.no_grad():
            logits_image, logits_caption = model(img_batch, captions_batch)
        labels = torch.arange(len(img_batch)).to(
            DEVICE
        )  # we are interested on predicting the right caption which is the caption position of every image
        total_loss = (
            loss_img(logits_image, labels) + loss_caption(logits_caption, labels)
        ) / 2
        test_loss += total_loss.item()

    test_total_loss = test_loss / len(test_dl)
    print(f"Validation Loss {test_total_loss:.3f}")
    return test_total_loss


results = train_model(N_EPOCHS, tr_dl, ts_dl)

# Plot loss curves

plt.plot(results["val_loss"], label="validation loss")
plt.plot(results["train_loss"], label="train loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.suptitle("Training & Validation Loss during fine-tuning")
plt.show()
