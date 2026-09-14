"""Deterministic channel compression interfaces."""

from .base import CompressionResult, Compressor
from .fixed_ratio import FixedRatioCompressor
from .identity import IdentityCompressor

__all__ = ["CompressionResult", "Compressor", "FixedRatioCompressor", "IdentityCompressor"]

