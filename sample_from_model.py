from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from PIL import Image

PATH = "118"
synth_path = "/remote/nas/algo/avim/SYNTHETIC_DATA/Diffusion_Synthetic_Samples_Exp_512_TOP"
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
    sampling_timesteps=100, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective='pred_v',
    beta_schedule='linear'
)

trainer = Trainer(
    diffusion,
    '/remote/nas/algo/avim/Diffusion/atix_explosives_v1_true',
    train_batch_size=1,
    train_lr=1e-4,
    train_num_steps=200000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True,                       # turn on mixed precision
    calculate_fid=True,              # whether to calculate fid during training
    num_samples=4,
    num_fid_samples=10,
    results_folder="./results_512_64_pred_v_atix60x40_explosive_material_top_view"
)
trainer.load(PATH)

# sampled_images = diffusion.sample(batch_size=10)
#
# batch_samples = sampled_images.cpu().numpy()
pathlib.Path(synth_path).mkdir(parents=True, exist_ok=True)

for i in range(0, 15):
    # plt.figure()
    #
    # plt.imshow(np.einsum('kij->ijk', batch_samples[i]))

    sampled_images = diffusion.sample(batch_size=5)

    batch_samples = sampled_images.cpu().numpy()

    for j in range(0, batch_samples.shape[0]):
        arr = 255.0*np.einsum('kij->ijk', batch_samples[j])
        im = Image.fromarray(arr.astype(np.uint8))

        im.save(os.path.join(synth_path, f"im_synth_exp{i}_{j}.tiff"))