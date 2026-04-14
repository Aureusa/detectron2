import sys
from pathlib import Path
from tqdm import tqdm

# Add detectron2 to path (assumes detectron2 is in the parent directory structure)
detectron2_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(detectron2_root))

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data import DatasetCatalog
from data.register_dataset import main as register_datasets


if __name__ == "__main__":
    register_datasets("/home/s4861264/detectron2/projects/LoTSS-GRG-detect/configs/dataset_configs/dataset_b2s_rgz_masked_rcnn.yaml")

    dataset_dicts = DatasetCatalog.get("b2s_train")
    repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dataset_dicts,
        repeat_thresh=0.3
    )

    SCS_CATEGORY_ID = 0
    MCS_CATEGORY_ID = 1

    print("Categories in dataset:")
    category_ids = set()
    for d in tqdm(dataset_dicts, desc="Processing dataset to find categories"):
        for a in d.get("annotations", []):
            category_ids.add(a["category_id"])
    print(f"Category IDs: {sorted(category_ids)}")

    print("Sorting MCS images...")
    mcs_images = [d for d in dataset_dicts if any(
        a["category_id"] == MCS_CATEGORY_ID 
        for a in d.get("annotations", [])
    )]
    print("Sorting SCS-only images...")
    scs_only_images = [d for d in dataset_dicts if all(
        a["category_id"] == SCS_CATEGORY_ID 
        for a in d.get("annotations", [])
    )]

    print(f"Total images: {len(dataset_dicts)}")
    print(f"MCS images: {len(mcs_images)} ({100*len(mcs_images)/len(dataset_dicts):.1f}%)")
    print(f"Max repeat factor: {repeat_factors.max():.2f}")
    mcs_indices = [i for i, d in enumerate(dataset_dicts) if any(
        a["category_id"] == MCS_CATEGORY_ID 
        for a in d.get("annotations", [])
    )]
    mean_repeat_factor_mcs = repeat_factors[mcs_indices].mean() if mcs_indices else 0
    print(f"Mean repeat factor for MCS images: {mean_repeat_factor_mcs:.2f}")
    print(f"Effective dataset size: {repeat_factors.sum():.0f} images per epoch")
