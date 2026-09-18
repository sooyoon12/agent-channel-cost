"""Deterministic channel compression interfaces."""

from .base import CompressionResult, Compressor
from .contract_aware import ContractAwareCompressor
from .fixed_ratio import FixedRatioCompressor
from .identity import IdentityCompressor

__all__ = [
    "CompressionResult",
    "Compressor",
    "ContractAwareCompressor",
    "FixedRatioCompressor",
    "IdentityCompressor",
]
