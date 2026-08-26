"""
training loops for phase one supervised pretraining and phase two
adaptation, plus the shared validation routine
"""
from .train import (
    pretrain_epoch,
    discriminator_step,
    adapt_epoch,
)
from .evaluate import validate_on_loader

__all__ = [
    "pretrain_epoch",
    "discriminator_step",
    "adapt_epoch",
    "validate_on_loader",
]
