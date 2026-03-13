from typing import List

import torch


class Validity:
	"""Stores proposal validity targets as an N-length tensor."""

	def __init__(self, tensor: torch.Tensor):
		if not isinstance(tensor, torch.Tensor):
			tensor = torch.as_tensor(tensor, dtype=torch.float32)
		else:
			tensor = tensor.to(torch.float32)

		if tensor.numel() == 0:
			tensor = tensor.reshape(0)
		if tensor.dim() != 1:
			raise ValueError(f"Validity expects a 1D tensor [N], got {tuple(tensor.shape)}")

		self.tensor = tensor

	def clone(self) -> "Validity":
		return Validity(self.tensor.clone())

	def to(self, device: torch.device) -> "Validity":
		return Validity(self.tensor.to(device=device))

	@property
	def device(self) -> torch.device:
		return self.tensor.device

	def __len__(self) -> int:
		return self.tensor.shape[0]

	def __getitem__(self, item) -> "Validity":
		t = self.tensor[item]
		if t.dim() == 0:
			t = t.unsqueeze(0)
		if t.dim() != 1:
			raise ValueError(f"Indexing Validity produced invalid shape {tuple(t.shape)}")
		return Validity(t)

	def __repr__(self) -> str:
		return f"Validity(shape={tuple(self.tensor.shape)})"

	@classmethod
	def cat(cls, validities: List["Validity"]) -> "Validity":
		if len(validities) == 0:
			return cls(torch.empty((0,), dtype=torch.float32))
		return cls(torch.cat([v.tensor for v in validities], dim=0))

