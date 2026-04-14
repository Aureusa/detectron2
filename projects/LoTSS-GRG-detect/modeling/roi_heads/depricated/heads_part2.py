

def build_physics_heads(cfg, input_dim) -> PhysicsAwareHeads:
    hidden_dim = cfg.MODEL.HEADS.HIDDEN_DIM
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    dropout = cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT
    decouple_validity_projection = cfg.MODEL.VALIDITY_HEAD.DECOUPLE_PROJECTION
    two_classes = cfg.MODEL.VALIDITY_HEAD.TWO_CLASSES # Whether to treat membership as binary (member vs non-member) or multi-class (MCS/SCS/invalid)
    return PhysicsAwareHeads(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        dropout=dropout,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
        decouple_validity_projection=decouple_validity_projection,
        two_classes=two_classes,
    )

def build_set_heads(cfg, proposal_input_dim, membership_input_dim) -> SetHeads:
    membership_head_hidden_dim = cfg.MODEL.MEMBERSHIP_HEAD.HIDDEN_DIM
    proposal_head_hidden_dim = cfg.MODEL.VALIDITY_HEAD.HIDDEN_DIM
    membership_loss_weight = cfg.MODEL.MEMBERSHIP_HEAD.LOSS_WEIGHT
    proposal_loss_weight = cfg.MODEL.VALIDITY_HEAD.LOSS_WEIGHT
    dropout = cfg.MODEL.MEMBERSHIP_HEAD.DROPOUT
    two_classes = cfg.MODEL.VALIDITY_HEAD.TWO_CLASSES # Whether to treat membership as binary (member vs non-member) or multi-class (MCS/SCS/invalid)
    return SetHeads(
        proposal_input_dim=proposal_input_dim,
        membership_input_dim=membership_input_dim,
        membership_head_hidden_dim=membership_head_hidden_dim,
        proposal_head_hidden_dim=proposal_head_hidden_dim,
        dropout=dropout,
        membership_loss_weight=membership_loss_weight,
        proposal_loss_weight=proposal_loss_weight,
        two_classes=two_classes,
    )
