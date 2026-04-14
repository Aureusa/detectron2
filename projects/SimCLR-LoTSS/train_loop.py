import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
import detectron2.model_zoo as mz
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import os

from simclr import SimCLRWithFPN
from dataloader import SimCLRCOCODataset
from nt_xent import nt_xent_loss
from register_dataset import main as register_dataset


def export_detectron2_backbone_checkpoint(backbone_state_dict, output_path):
    torch.save(
        {
            "model": {f"backbone.{key}": value for key, value in backbone_state_dict.items()},
            "__author__": "SimCLR-LoTSS",
        },
        output_path,
    )

class RadioChannelAugment:
    """
    Physically meaningful channel augmentation for 3-channel radio images.
    Channels: [3σ clip, 5σ clip, sqrt stretch]
    """
    def __call__(self, x):  # x: (3, H, W) tensor
        # Simulate noise variation: add small Gaussian noise per channel
        # scaled to the channel's dynamic range
        if torch.rand(1) < 0.5:
            noise_scale = 0.02  # 2% noise — realistic calibration uncertainty
            x = x + torch.randn_like(x) * noise_scale * x.std(dim=(-2,-1), keepdim=True)
        
        # Simulate flux calibration uncertainty: global intensity scaling
        # (all channels scaled together to preserve relative channel ratios)
        if torch.rand(1) < 0.5:
            scale = torch.FloatTensor(1).uniform_(0.85, 1.15)
            x = x * scale
        
        return x

def add_simclr_config(cfg):
    cfg.SIMCLR = CN()
    cfg.SIMCLR.PROJECTION_DIM = 128
    cfg.SIMCLR.TEMPERATURE = 0.5
    cfg.SIMCLR.NUM_EPOCHS = 200
    cfg.SIMCLR.WARMUP_EPOCHS = 10
    cfg.SIMCLR.BATCH_SIZE = 256
    cfg.SIMCLR.BASE_LR = 3e-4
    cfg.SIMCLR.WEIGHT_DECAY = 1e-4
    cfg.SIMCLR.CHECKPOINT_PERIOD = 50
    cfg.SIMCLR.AUG = CN()
    cfg.SIMCLR.AUG.CROP_SCALE_MIN = 0.2
    cfg.SIMCLR.AUG.CROP_SCALE_MAX = 1.0
    cfg.SIMCLR.AUG.COLOR_JITTER_PROB = 0.8
    cfg.SIMCLR.AUG.GRAYSCALE_PROB = 0.2
    cfg.SIMCLR.AUG.BLUR_PROB = 0.5

def setup_cfg(args):
    cfg = get_cfg()
    add_simclr_config(cfg)                    # register custom keys first
    cfg.merge_from_file(args.config_file)     # then load yaml
    cfg.freeze()
    return cfg

def compute_dataset_stats(cfg, dataset_name, num_samples=10000):
    from detectron2.data import DatasetCatalog
    import numpy as np
    from PIL import Image
    from detectron2.data import detection_utils as utils
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    sample = np.random.choice(
        dataset_dicts, 
        min(num_samples, len(dataset_dicts)), 
        replace=False
    )
    
    means = []
    stds = []
    for d in tqdm(sample, desc="Computing dataset stats"):
        img = utils.read_image(d["file_name"], format=cfg.INPUT.FORMAT)
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    
    mean = np.array(means).mean(axis=0)
    std = np.array(stds).mean(axis=0)
    print(f"Channel means (RGB): {mean}")
    print(f"Channel stds  (RGB): {std}")
    return mean, std

def build_transform(pixel_mean, pixel_std):
    """Apply same preprocessing as downstream Detectron2: load format, [0,255] scale, then normalize with cfg stats."""
    return T.Compose([
        # Convert HWC uint8 in cfg.INPUT.FORMAT -> CHW float tensor in [0, 255].
        T.Lambda(lambda x: torch.from_numpy(x.copy()).permute(2, 0, 1).float()),
        # Spatial — all physically valid for radio
        T.RandomResizedCrop(224, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomChoice([
            T.RandomRotation(degrees=(0, 0)),
            T.RandomRotation(degrees=(90, 90)),
            T.RandomRotation(degrees=(180, 180)),
            T.RandomRotation(degrees=(270, 270)),
        ]),
        # Beam size simulation
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.5, 3.0))], p=0.5),
        # Radio-specific intensity augmentation
        RadioChannelAugment(),
        # Normalize using downstream model's PIXEL_MEAN and PIXEL_STD
        T.Lambda(lambda x: (x - pixel_mean) / pixel_std),
    ])

def train(cfg):
    register_dataset()
    
    # ── Backbone ──────────────────────────────────────────────────────────────
    d2_model = build_model(cfg)       # loads MODEL.WEIGHTS automatically
    backbone = d2_model.backbone

    simclr_model = SimCLRWithFPN(
        backbone,
        projection_dim=cfg.SIMCLR.PROJECTION_DIM
    ).cuda()

    # ── Dataloader ────────────────────────────────────────────────────────────
    # Compute stats ONCE at startup
    mean, std = compute_dataset_stats(cfg, cfg.DATASETS.TRAIN[0], num_samples=10000)

    print("Using dataset normalization stats:")
    print(f"INPUT.PIXEL_MEAN: {mean}")
    print(f"INPUT.PIXEL_STD: {std}")
    
    # Convert to tensors
    pixel_mean = torch.from_numpy(mean).float().view(3, 1, 1)
    pixel_std = torch.from_numpy(std).float().view(3, 1, 1)

    transform = build_transform(pixel_mean, pixel_std)
    dataset_name = cfg.DATASETS.TRAIN[0]
    dataset = SimCLRCOCODataset(cfg, dataset_name, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.SIMCLR.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": simclr_model.backbone.parameters(), "lr": cfg.SIMCLR.BASE_LR * 0.1},
        {"params": simclr_model.projector.parameters(), "lr": cfg.SIMCLR.BASE_LR},
    ], weight_decay=cfg.SIMCLR.WEIGHT_DECAY)

    num_epochs = cfg.SIMCLR.NUM_EPOCHS
    warmup_steps = cfg.SIMCLR.WARMUP_EPOCHS * len(dataloader)
    total_steps = num_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device="cuda")

    # ── Resume if checkpoint exists ───────────────────────────────────────────
    start_epoch = 0
    best_ckpt = os.path.join(cfg.OUTPUT_DIR, "simclr_best.pth")
    best_d2_ckpt = os.path.join(cfg.OUTPUT_DIR, "simclr_best_detectron2.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt)
        simclr_model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        simclr_model.projector.load_state_dict(ckpt["projector_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    best_loss = float('inf')

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        simclr_model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x1, x2 in pbar:
            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                z1 = simclr_model(x1)
                z2 = simclr_model(x2)
                loss = nt_xent_loss(z1, z2, cfg.SIMCLR.TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(simclr_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | avg loss: {avg_loss:.4f}")

        # Write the avg loss to a file for later plotting
        with open(os.path.join(cfg.OUTPUT_DIR, "training_log.txt"), "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            backbone_state_dict = simclr_model.backbone.state_dict()
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": backbone_state_dict,
                "projector_state_dict": simclr_model.projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, best_ckpt)
            export_detectron2_backbone_checkpoint(backbone_state_dict, best_d2_ckpt)

        if (epoch + 1) % cfg.SIMCLR.CHECKPOINT_PERIOD == 0:
            backbone_state_dict = simclr_model.backbone.state_dict()
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": backbone_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(cfg.OUTPUT_DIR, f"simclr_epoch{epoch+1}.pth"))
            export_detectron2_backbone_checkpoint(
                backbone_state_dict,
                os.path.join(cfg.OUTPUT_DIR, f"simclr_epoch{epoch+1}_detectron2.pth"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    train(cfg)
    