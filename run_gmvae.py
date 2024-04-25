import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.gmvae import GMVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'gmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}',  args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
gmvae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=gmvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=args.train == 2)

else:
    ut.load_model_by_name(gmvae, global_step=args.iter_max)
    ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=True)
    # x_samples = gmvae.sample_x(200)
    # np_x_samples = x_samples.detach().cpu().numpy().reshape(-1, 28, 28)

    # # Create a grid of 10x20
    # fig, axs = plt.subplots(10, 20, figsize=(10, 8))
    # fig.subplots_adjust(hspace = 0, wspace = 0)
    # for ax in axs.ravel():
    #     ax.axis('off')
    # for i, ax in enumerate(axs.ravel()):
    #     ax.imshow(np_x_samples[i], cmap='gray',aspect='auto')
    # plt.savefig("visualize_200_digit_gmvae.png")
    # plt.show()
