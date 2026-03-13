from typing import List

import torch


class Membership:
	"""Stores proposal-component membership targets as an NxC tensor."""

	def __init__(self, tensor: torch.Tensor):
		if not isinstance(tensor, torch.Tensor):
			tensor = torch.as_tensor(tensor, dtype=torch.float32)
		else:
			tensor = tensor.to(torch.float32)

		if tensor.numel() == 0:
			tensor = tensor.reshape(0, 0)
		if tensor.dim() != 2:
			raise ValueError(f"Membership expects a 2D tensor [N, C], got {tuple(tensor.shape)}")

		self.tensor = tensor

	def clone(self) -> "Membership":
		return Membership(self.tensor.clone())

	def to(self, device: torch.device) -> "Membership":
		return Membership(self.tensor.to(device=device))

	@property
	def device(self) -> torch.device:
		return self.tensor.device

	def __len__(self) -> int:
		return self.tensor.shape[0]

	def __getitem__(self, item) -> "Membership":
		t = self.tensor[item]
		if t.dim() == 1:
			t = t.unsqueeze(0)
		if t.dim() != 2:
			raise ValueError(f"Indexing Membership produced invalid shape {tuple(t.shape)}")
		return Membership(t)

	def __repr__(self) -> str:
		return f"Membership(shape={tuple(self.tensor.shape)})"

	@classmethod
	def cat(cls, memberships: List["Membership"]) -> "Membership":
		if len(memberships) == 0:
			return cls(torch.empty((0, 0), dtype=torch.float32))
		return cls(torch.cat([m.tensor for m in memberships], dim=0))

