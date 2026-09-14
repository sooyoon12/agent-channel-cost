"""Tokenizer adapters with explicit exact/proxy provenance."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from typing import Any, Protocol


class Tokenizer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def is_exact_for_model(self) -> bool: ...

    def count_text(self, text: str) -> int: ...

    def count_json(self, value: Any) -> int: ...


@dataclass(frozen=True)
class TokenizerInfo:
    model_id: str
    name: str
    version: str
    is_exact_for_model: bool
    limitation: str | None


class TiktokenTokenizer:
    def __init__(self, encoding: Any, info: TokenizerInfo) -> None:
        self._encoding = encoding
        self.info = info

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def version(self) -> str:
        return self.info.version

    @property
    def is_exact_for_model(self) -> bool:
        return self.info.is_exact_for_model

    def count_text(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_json(self, value: Any) -> int:
        return self.count_text(canonical_json(value))


def canonical_json(value: Any) -> str:
    """Stable local serialization for content-token accounting.

    This is deliberately not claimed to reproduce provider protocol framing.
    The difference belongs in protocol residual.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolve_tiktoken(model_id: str, proxy_encoding: str | None = None) -> TiktokenTokenizer:
    """Resolve an exact public model encoding or an explicitly labeled proxy.

    No network or model API call is made. ImportError is left actionable rather
    than silently replacing token counts with characters or bytes.
    """

    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is not installed; install the optional 'tokenizer' dependency "
            "before measuring realized_ratio"
        ) from exc

    try:
        version = importlib.metadata.version("tiktoken")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError:
        if proxy_encoding is None:
            raise RuntimeError(
                f"tiktoken has no exact mapping for {model_id!r}; select and record a proxy encoding"
            ) from None
        encoding = tiktoken.get_encoding(proxy_encoding)
        return TiktokenTokenizer(
            encoding,
            TokenizerInfo(
                model_id=model_id,
                name=proxy_encoding,
                version=version,
                is_exact_for_model=False,
                limitation=f"proxy encoding; tiktoken has no mapping for {model_id!r}",
            ),
        )

    return TiktokenTokenizer(
        encoding,
        TokenizerInfo(
            model_id=model_id,
            name=encoding.name,
            version=version,
            is_exact_for_model=True,
            limitation=None,
        ),
    )
