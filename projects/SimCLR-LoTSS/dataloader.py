from detectron2.data import DatasetCatalog
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch

from detectron2.data import detection_utils as utils


class SimCLRCOCODataset(Dataset):
    def __init__(self, cfg, dataset_name, transform):
        # DatasetCatalog returns list of dicts with file_name, annotations, etc.
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.transform = transform
        self.input_format = cfg.INPUT.FORMAT

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        record = self.dataset_dicts[idx]
        image = utils.read_image(record["file_name"], format=self.input_format)
        
        # Apply augmentation twice → two views
        x1 = self.transform(image)
        x2 = self.transform(image)
        return x1, x2
    

def get_simclr_dataloader(dataset_name, batch_size=256, num_workers=8):
    # SimCLR augmentation pipeline
    simclr_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    
    dataset = SimCLRCOCODataset(dataset_name, simclr_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # important for NT-Xent loss
    )
