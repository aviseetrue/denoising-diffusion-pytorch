from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
import numpy as np

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,           # number of steps
    sampling_timesteps=250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/remote/nas/algo/avim/Diffusion/ankara_sponges_guns',
    train_batch_size=32,
    train_lr = 1e-4,
    train_num_steps=10000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid=True,              # whether to calculate fid during training
    save_and_sample_every=1000,
    results_folder="./results_avi"
)

trainer.train()
sampled_images = diffusion.sample(batch_size=10)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),
                            interpolation='nearest')
    plt.show()

show(sampled_images[1].cpu())