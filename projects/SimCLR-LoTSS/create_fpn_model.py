import argparse

import torch


def _to_detectron2_backbone_state_dict(backbone_state_dict):
    return {f"backbone.{key}": value for key, value in backbone_state_dict.items()}


def create_fpn_model(model_pth: str, backbone_pth: str):
    checkpoint = torch.load(model_pth, map_location="cpu")

    if "backbone_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{model_pth}' does not contain 'backbone_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    backbone_state_dict = checkpoint["backbone_state_dict"]
    detectron2_state_dict = _to_detectron2_backbone_state_dict(backbone_state_dict)

    torch.save(
        {
            "model": detectron2_state_dict,
            "__author__": "SimCLR-LoTSS",
        },
        backbone_pth,
    )

    print(f"Saved Detectron2-compatible backbone checkpoint to {backbone_pth}")
    print("Example keys:")
    for key in list(detectron2_state_dict.keys())[:5]:
        print(f"  {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pth", help="Path to the SimCLR checkpoint containing backbone_state_dict")
    parser.add_argument("backbone_pth", help="Path to save the Detectron2-compatible backbone checkpoint")
    args = parser.parse_args()

    create_fpn_model(args.model_pth, args.backbone_pth)
    