import random

import numpy as np
import torch

patch_size = (128, 128, 128)
batch_size = 2
num_workers = 2
seed = 42

sup_epochs = 50
adapt_epochs = 200

sup_lr = 1e-3
sup_step_size = 5
sup_gamma = 0.9
rad_weight_early = 0.1
rad_weight_late = 0.2
rad_weight_switch_epoch = 5

encoder_lr = 1e-4
decoder_lr = 1e-3
disc_lr = 1e-4
disc_betas = (0.5, 0.999)
adapt_step_size = 5
adapt_gamma = 0.95
weight_decay = 1e-5
clip_grad = 5.0

grl_alpha_start = 1.0
grl_alpha_end = 2.0

rad_cons_warmup_epochs = 10
lambda_rad_cons = 0.0001
rad_h_weight = 0.2

lambda_adv = 0.001
lambda_coral = 0.001

self_feedback_beta_max = 1.0
self_feedback_e_start = 10
self_feedback_e_end = adapt_epochs

rad_feats_save = "hgg_radiomics.npz"
ckpt_dir = "checkpoints"
fig_dir = "figures"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(value=None):
    if value is None:
        value = seed

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
