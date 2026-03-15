# e.g. projects/LoTSS-GRG-detect/config/custom_config.py
from detectron2.config import CfgNode as CN

def add_lotss_grg_config(cfg):
    # Physics Feature Attention Network (FAN) configuration
    cfg.MODEL.PHYSICS_FAN = CN()
    cfg.MODEL.PHYSICS_FAN.NUM_COMPONENTS = 10
    cfg.MODEL.PHYSICS_FAN.NUM_PHYSICS_FEATURES = 64
    cfg.MODEL.PHYSICS_FAN.EMBEDDING_DIM = 128
    cfg.MODEL.PHYSICS_FAN.EMBEDDING_DROPOUT = 0.0
    cfg.MODEL.PHYSICS_FAN.NUM_ATTENTION_HEADS = 8
    cfg.MODEL.PHYSICS_FAN.ATTENTION_DROPOUT = 0.0

    # Fusion module configuration
    cfg.MODEL.FUSION_MODULE = CN()
    cfg.MODEL.FUSION_MODULE.TYPE = "AttentionFusionModule"
    cfg.MODEL.FUSION_MODULE.DROPOUT = 0.0
    cfg.MODEL.FUSION_MODULE.NUM_HEADS = 8

    # ROI Align configuration
    cfg.MODEL.ROI_ALIGN = CN()
    cfg.MODEL.ROI_ALIGN.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_ALIGN.POOLER_RESOLUTION = 28
    cfg.MODEL.ROI_ALIGN.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_ALIGN.POOLER_TYPE = "ROIAlignV2"

    # Custom heads configuration
    # Validity head for proposal classification
    cfg.MODEL.VALIDITY_HEAD = CN()
    cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.VALIDITY_HEAD.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.VALIDITY_HEAD.DECOUPLE_PROJECTION = True

    # Membership head for component classification
    cfg.MODEL.MEMBERSHIP_HEAD = CN()
    cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM = 256
    cfg.MODEL.MEMBERSHIP_HEAD.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MEMBERSHIP_HEAD.NUM_HEADS = 4
    cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT = 0.0

    # Shared head configuration (if needed for future use)
    cfg.MODEL.HEADS = CN()
    cfg.MODEL.HEADS.HIDDEN_DIM = 256   # shared projection output dim
    cfg.MODEL.HEADS.NUM_HEADS = 4      # ComponentAttention heads
    cfg.MODEL.HEADS.DROPOUT = 0.0

    cfg.TRAINER = CN()
    cfg.TRAINER.NAME = "GRGTrainer"

    cfg.DATASETS = CN()
    cfg.DATASETS.POSITIVE_FRACTION = 0.20  # Fraction of positive samples in each batch pos:neg - 1:5
    cfg.DATASETS.TRAIN = ("",)
    cfg.DATASETS.TEST = ("",)
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 100000
    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 100000
    