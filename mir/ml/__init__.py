"""Neural codecs and density methods (Part 2). Requires the ``[ml]`` extra (PyTorch).

* forward codec — sequence → (compact) embedding (:mod:`mir.ml.encoder`, :mod:`mir.ml.train`);
* inverse codec — embedding → sequence (:mod:`mir.ml.decoder`);
* Pgen regressor — sequence → log10 Pgen (:mod:`mir.ml.train`);
* density methods — planned.

Torch is imported lazily: ``import mir.ml`` (and ``import mir``) work without torch;
accessing a codec symbol pulls it in.
"""

from mir.ml.tokenize import encode_indices, encode_onehot  # pure numpy, always safe

__all__ = [
    "encode_onehot",
    "encode_indices",
    "SequenceEncoder",
    "SequenceDecoder",
    "ForwardEncoder",
    "InverseDecoder",
    "PgenRegressor",
    "train_forward_encoder",
    "train_inverse_decoder",
    "train_pgen_regressor",
]

_TRAIN = {"ForwardEncoder", "InverseDecoder", "PgenRegressor",
          "train_forward_encoder", "train_inverse_decoder", "train_pgen_regressor"}


def __getattr__(name: str):
    if name == "SequenceEncoder":
        from mir.ml.encoder import SequenceEncoder

        return SequenceEncoder
    if name == "SequenceDecoder":
        from mir.ml.decoder import SequenceDecoder

        return SequenceDecoder
    if name in _TRAIN:
        from mir.ml import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
