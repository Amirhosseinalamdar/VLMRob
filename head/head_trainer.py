#!/usr/bin/env python3
import os
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import open_clip
import random
import torch, random, numpy as np
import collections, random
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. Dictionary: visual encoders available
# ==========================================================
VLM_MODELS = {
    "llava": ("ViT-L-14", "laion2b_s32b_b82k"),  # LLaVA 1.5/1.6 visual tower
    # "clip_laionL14": ("ViT-L-14", "laion2b_s32b_b82k"),
    # "clip_openaiB32": ("ViT-B-32", "openai"),
}

# ==========================================================
# 2. Model Components
# ==========================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def build_encoders(vlm_name, target_model, device):
    if vlm_name not in VLM_MODELS:
        raise ValueError(f"Unknown VLM '{vlm_name}'. Available: {list(VLM_MODELS.keys())}")
    img_encoder_model, img_encoder_weights = VLM_MODELS[vlm_name]
    print(f"Loading image encoder for '{vlm_name}' -> ({img_encoder_model}, {img_encoder_weights})")
    llava_encoder, _, _ = open_clip.create_model_and_transforms(
        img_encoder_model, pretrained=img_encoder_weights
    )
    print(f"Loading target CLIP encoder -> ({target_model[0]}, {target_model[1]})")
    clip_encoder, _, _ = open_clip.create_model_and_transforms(
        target_model[0], pretrained=target_model[1]
    )
    llava_encoder = llava_encoder.visual.eval().to(device)
    clip_encoder = clip_encoder.visual.eval().to(device)
    for p in llava_encoder.parameters():
        p.requires_grad = False
    for p in clip_encoder.parameters():
        p.requires_grad = False
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        dim_a = llava_encoder(dummy).shape[-1]
        dim_b = clip_encoder(dummy).shape[-1]
    print(f"Encoder dims -> source: {dim_a}, target: {dim_b}")
    return llava_encoder, clip_encoder, dim_a, dim_b

# ==========================================================
# 3. Dataset
# ==========================================================
def get_dataloaders(dataset_name, imagenet_root="./data"):
    if dataset_name == "cifar10":
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root="../cifar10/data", train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root="../cifar10/data", train=False, download=False, transform=transform)
    elif dataset_name == "imagenet1k":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])
        print('building dataset ...')
        fullset = torchvision.datasets.ImageFolder(root=imagenet_root, transform=transform)
        targets = [lbl for _, lbl in fullset.samples]  # labels from ImageFolder
        train_idx, test_idx = train_test_split(
            list(range(len(targets))),
            test_size=1000,         # absolute number or fraction
            stratify=targets,       # ensures per‑class proportion or count
            random_state=42
        )
        trainset = torch.utils.data.Subset(fullset, train_idx)
        testset  = torch.utils.data.Subset(fullset, test_idx)
        # # Build (class → indices) mapping
        # by_class = collections.defaultdict(list)
        # for idx, (_, label) in enumerate(fullset):
        #     by_class[label].append(idx)
        # # Random pick 3 from each class for test, rest for train
        # train_indices, test_indices = [], []
        # random.seed(42)
        # for label, idxs in by_class.items():
        #     if len(idxs) >= 3:
        #         random.shuffle(idxs)
        #         test_indices.extend(idxs[:3])
        #         train_indices.extend(idxs[3:])
        #     else:
        #         train_indices.extend(idxs)
        # trainset = torch.utils.data.Subset(fullset, train_indices)
        # testset = torch.utils.data.Subset(fullset, test_indices)
        print('dataloaders built')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    return trainloader, testloader

# ==========================================================
# 4. Evaluation
# ==========================================================
def evaluate_alignment(model, enc_a, enc_b, loader, device):
    model.eval()
    sims = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            emb_a = enc_a(imgs)
            emb_b = enc_b(imgs)
            mapped = model(emb_a)
            mapped = F.normalize(mapped, dim=-1)
            emb_b = F.normalize(emb_b, dim=-1)
            sims.append((mapped * emb_b).sum(dim=-1).cpu())
    return torch.cat(sims).mean().item()

# ==========================================================
# 5. Training loop (now includes per‑epoch evaluation)
# ==========================================================
def train(model, enc_a, enc_b, train_loader, test_loader, optimizer, criterion,
          epochs, device, dataset_name):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            with torch.no_grad():
                emb_a = enc_a(imgs)
                emb_b = enc_b(imgs)
            pred_b = model(emb_a)
            loss = criterion(pred_b, emb_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             avg=f"{running_loss/(pbar.n+1):.4f}")
        mean_loss = running_loss / len(train_loader)
        # Evaluate after every epoch
        score = evaluate_alignment(model, enc_a, enc_b, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Mean loss: {mean_loss:.6f} | "
              f"Alignment on {dataset_name}: {score:.4f}")
def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ==========================================================
# 6. Main
# ==========================================================
def main(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_clip = ("ViT-B-32", "openai")
    enc_a, enc_b, dim_a, dim_b = build_encoders(args.vlm, target_clip, device)
    mlp = ProjectionHead(dim_a, dim_b).to(device)
    train_loader, test_loader = get_dataloaders(args.dataset, args.imagenet_root)
    optimizer = optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()
    train(mlp, enc_a, enc_b, train_loader, test_loader,
          optimizer, criterion, args.epochs, device, args.dataset)
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/mlp_{args.vlm}_{args.dataset}_lr{args.lr}_wd{args.wd}.pth"
    torch.save(mlp.state_dict(), save_path)
    print(f"✅ Saved weights to: {save_path}")

# ==========================================================
# 7. Entry
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and align VLM and CLIP encoders")
    parser.add_argument("--vlm", type=str, default="llava", choices=list(VLM_MODELS.keys()),
                        help="which VLM encoder to use")
    parser.add_argument("--dataset", type=str, default="imagenet1k",
                        choices=["cifar10", "imagenet1k"],
                        help="dataset to use for training and eval")
    parser.add_argument("--imagenet_root", type=str, default="../data/imagenet/val",
                        help="path to ImageNet root")   
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    args = parser.parse_args()
    main(args)