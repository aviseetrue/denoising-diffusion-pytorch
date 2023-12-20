from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion, Trainer

import matplotlib.pyplot as plt
import numpy as np

PATH = "14"

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True,
    resnet_block_groups=2
)

diffusion = GaussianDiffusion(
    model,
    image_size=512,
    timesteps=400,           # number of steps
    sampling_timesteps=400, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective='pred_v',
    beta_schedule='linear'
)

trainer = Trainer(
    diffusion,
    '/remote/nas/algo/avim/Diffusion/atix_bullets/bullets_atix_crops_jpg',
    train_batch_size=1,
    train_lr=1e-4,
    train_num_steps=200000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True,                       # turn on mixed precision
    calculate_fid=True,              # whether to calculate fid during training
    num_samples=4,
    num_fid_samples=10,
    results_folder="./results_512_64_pred_v_atix60x40_bullets_top_view"
)

# trainer.load(PATH)

trainer.train()

# sampled_images = diffusion.sample(batch_size=15)
#
# batch_samples = sampled_images.cpu().numpy()
#
# for i in range(0, 15):
#     plt.figure()
#
#     plt.imshow(np.einsum('kij->ijk', batch_samples[i]))