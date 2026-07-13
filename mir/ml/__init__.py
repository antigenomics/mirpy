"""Neural codecs and density methods (Part 2). Requires the ``[ml]`` extra (PyTorch).

* forward codec — sequence → embedding (:mod:`mir.ml.encoder`, :mod:`mir.ml.train`);
* inverse decoder, Pgen-from-embedding regressor, density methods — planned.

Torch is imported lazily: ``import mir.ml`` (and ``import mir``) work without torch;
accessing a codec symbol pulls it in.
"""

from mir.ml.tokenize import encode_indices, encode_onehot  # pure numpy, always safe

__all__ = [
    "encode_onehot",
    "encode_indices",
    "SequenceEncoder",
    "ForwardEncoder",
    "train_forward_encoder",
]


def __getattr__(name: str):
    if name == "SequenceEncoder":
        from mir.ml.encoder import SequenceEncoder

        return SequenceEncoder
    if name in ("ForwardEncoder", "train_forward_encoder"):
        from mir.ml import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
