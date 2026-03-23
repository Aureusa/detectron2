import torch
from torch import nn

from detectron2.config import configurable

from ..vanila import TransformerBlock, PhysicsFeatureEmbedder


class PhysicsFAN(nn.Module):
    """Physics Feature Attention Network (FAN) module for enhancing features with physics-based reasoning."""
    @configurable
    def __init__(
        self,
        num_components: int,
        num_physics_features: int,
        embedding_dropout: float,
        num_attention_heads: int,
        attention_dropout: float,
        embedding_dim: int,
        ):
        super().__init__()
        self.num_components = num_components
        self.num_physics_features = num_physics_features
        self.embedding_dim = embedding_dim

        self.physics_feature_embedder = PhysicsFeatureEmbedder(
            input_dim=num_physics_features,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            dropout=embedding_dropout
        )
        
        self.embedding_norm = nn.LayerNorm(embedding_dim)

        self.transformer_block = TransformerBlock(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout
        )

        # self.component_embedding = nn.Embedding(num_components, embedding_dim)  # Learnable embeddings for each component type

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_components": cfg.MODEL.PHYSICS_FAN.NUM_COMPONENTS,
            "num_physics_features": cfg.MODEL.PHYSICS_FAN.NUM_PHYSICS_FEATURES,
            "embedding_dropout": cfg.MODEL.PHYSICS_FAN.EMBEDDING_DROPOUT,
            "num_attention_heads": cfg.MODEL.PHYSICS_FAN.NUM_ATTENTION_HEADS,
            "attention_dropout": cfg.MODEL.PHYSICS_FAN.ATTENTION_DROPOUT,
            "embedding_dim": cfg.MODEL.PHYSICS_FAN.EMBEDDING_DIM,
        }

    def forward(self, features):
        """
        The forward method takes in physics features for
        each component and applies attention to enhance them.
        
        :param features: A list of physics feature tensors for each component, each of shape (P, C, num_physics_features).
        This should be a list of batched per proposal per component features, where P is the number of proposals,
        C is the number of components, and num_physics_features is the size of the physics feature vector for each component.
            TODO: Define how the input features are structured and how they are obtained from the data.
        :return: A list of enhanced feature tensors for each component, each of shape (P, C, embedding_dim)
        """
        # Binary membership matrix: (B, P, C), 1 if component is in proposal, 0 otherwise
        membership_matrix = self._binary_membership_matrix(features)  # (B, P, C)

        features = self._preprocess_features(features)  # Pre-process the input features (B, P, C, num_physics_features)

        # Add component id embeddings to the features to provide explicit component type information to the model.
        # _, _, C, _ = features.shape
        # component_ids = torch.arange(C, device=features.device)  # (C,)
        # component_embeds = self.component_embedding(component_ids)  # (C, embedding_dim)
        # component_embeds = component_embeds.unsqueeze(0).unsqueeze(0)  # (1, 1, C, embedding_dim) -> broadcast to (B, P, C, embedding_dim)

        embedded_features = self.physics_feature_embedder(features)  # (B, P, C, embedding_dim) - Batch size, P proposals, C components, embedding_dim features
        # embedded_features += component_embeds  # Add component type embeddings

        embedded_features = self.embedding_norm(embedded_features)  # Apply layer normalization

        unsq_membership_matrix = membership_matrix.unsqueeze(-1).float()  # (B, P, C, 1)
        embedded_features *= unsq_membership_matrix  # Mask out features for components not in the proposal

        attended_features, attn_scores = self.transformer_block(
            embedded_features,
            embedded_features,
            embedded_features,
            key_padding_mask=~membership_matrix.bool(),
            output_mask=membership_matrix.bool(),
        )  # Self-attention across components (B, P, C, embedding_dim)
        return {
            "attention_features": attended_features,
            "membership_matrix": membership_matrix,
            "attention_scores": attn_scores
        }

    def _preprocess_features(self, features):
        """
        Pre-process the input physics features for each component.
        This may involve normalizing the features, applying a linear transformation, etc.
        Also unpacking the features in a meaningful way to be used by the network.
        
        :param features: A list of physics feature tensors for each component, each of shape (P, C, num_physics_features).
        :return: A list of processed feature tensors for each component, each of shape (P, C, embedding_dim)
        """
        # features contains list of Instances with a field component_features of shape
        # (P, C, num_physics_features)
        # We are batching the features, so we need to stack them into a tensor of shape
        # (B, P, C, num_physics_features)
        # The features are already tensors on the right device
        physical_features = torch.stack([feat.component_features for feat in features], dim=0)  # (B, P, C, num_physics_features)
        return physical_features
    
    def _binary_membership_matrix(self, features):
        """
        Create a binary membership matrix indicating which components are present in each proposal.
        
        :param features: A list of physics feature tensors for each component, each of shape (P, C, num_physics_features).
        :return: A binary membership matrix of shape (B, P, C), where B is the batch size, P is the number of proposals, and C is the number of components.
        """
        # features contain list of Instances with a field component_mask of shape (P, C)
        # We are batching the features, so we need to stack them into a tensor of shape (B, P, C)
        # The features are already tensors on the right device
        membership_matrix = torch.stack([feat.component_mask for feat in features], dim=0)  # (B, P, C)
        return membership_matrix

def build_physics_fan(cfg):
    return PhysicsFAN(**PhysicsFAN.from_config(cfg))
