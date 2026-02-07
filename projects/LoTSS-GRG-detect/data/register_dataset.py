from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.structures import BoxMode
import yaml
import os
import logging

logger = logging.getLogger("LoTSS-GRG-detect.data.register_dataset")

def load_coco_json_xyxy(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load COCO format JSON with XYXY bboxes instead of XYWH.
    This is a wrapper around load_coco_json that fixes bbox_mode.
    """
    # Load using standard loader
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name, extra_annotation_keys)
    
    # Fix bbox_mode for all annotations
    for dataset_dict in dataset_dicts:
        for anno in dataset_dict.get("annotations", []):
            # Change from XYWH_ABS (which load_coco_json sets) to XYXY_ABS
            anno["bbox_mode"] = BoxMode.XYXY_ABS
    
    return dataset_dicts

def register_dataset(dataset: dict, dataset_name: str, data_root: str):
    split_name = dataset.get('NAME', 'default') # train, val, test
    dataset_name = f"{dataset_name}_{split_name}"
    
    ann_file = os.path.join(data_root, split_name, dataset.get('ANNOTATIONS', ''))
    img_dir = os.path.join(data_root, split_name, dataset.get('IMAGES', 'images'))
    
    logger.info(f"Registering dataset: {dataset_name}")
    logger.info(f"  Annotations: {ann_file}")
    logger.info(f"  Images dir: {img_dir}")
    logger.info(f"  BBox format: XYXY_ABS")
    
    # Register with custom loader that handles XYXY bboxes
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json_xyxy(ann_file, img_dir, dataset_name)
    )
    
    # Set metadata
    MetadataCatalog.get(dataset_name).set(
        json_file=ann_file,
        image_root=img_dir,
        evaluator_type="coco",
        thing_classes=["GRG"]
    )
    
    logger.info(f"Successfully registered dataset: {dataset_name}")
    return dataset_name

def main(cfg_filepath: str):
    # Load dataset configuration from YAML file
    logger.info(f"Loading dataset configuration from: {cfg_filepath}")
    with open(cfg_filepath, 'r') as f:
        dataset_cfg = yaml.safe_load(f)

    # Get the directory of the config file to resolve relative paths
    data_root = dataset_cfg.get('DATA_ROOT', '')
    logger.info(f"Data root directory: {data_root}")

    # Get dataset name
    dataset_name = dataset_cfg.get('DATASET_NAME', 'custom_dataset')
    logger.info(f"Dataset base name: {dataset_name}")

    # Get the train, val and test dataset configurations
    train_dataset = dataset_cfg.get('TRAIN', {})
    val_dataset = dataset_cfg.get('VALIDATION', {})
    test_dataset = dataset_cfg.get('TEST', {})

    registered_datasets = []
    
    if train_dataset.get('EXISTS', False):
        logger.info("Registering training dataset...")
        name = register_dataset(train_dataset, dataset_name, data_root)
        registered_datasets.append(name)
    else:
        logger.warning("Training dataset not configured or does not exist")
        
    if val_dataset.get('EXISTS', False):
        logger.info("Registering validation dataset...")
        name = register_dataset(val_dataset, dataset_name, data_root)
        registered_datasets.append(name)
    else:
        logger.warning("Validation dataset not configured or does not exist")
        
    if test_dataset.get('EXISTS', False):
        logger.info("Registering test dataset...")
        name = register_dataset(test_dataset, dataset_name, data_root)
        registered_datasets.append(name)
    else:
        logger.warning("Test dataset not configured or does not exist")
    
    logger.info(f"Dataset registration complete. Registered datasets: {registered_datasets}")
    return registered_datasets

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register datasets from YAML config")
    parser.add_argument(
        "--cfg",
        required=True,
        help="Path to the dataset configuration YAML file"
    )
    args = parser.parse_args()

    main(args.cfg)
