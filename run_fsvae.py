import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.fsvae import FSVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=990000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000,   help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,       help="Run ID. In case you want to run replicates")
args = parser.parse_args()
layout = [
    ('model={:s}',  'fsvae'),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_loader, labeled_subset, test_set = ut.get_svhn_data(device)
fsvae = FSVAE(name=model_name).to(device)
# writer = ut.prepare_writer(model_name, overwrite_existing=False)

# train(model=fsvae,
#       train_loader=train_loader,
#       labeled_subset=labeled_subset,
#       device=device,
#       y_status='fullsup',
#       tqdm=tqdm.tqdm,
#       writer=writer,
#       iter_max=args.iter_max,
#       iter_save=args.iter_save)
ut.load_model_by_name(fsvae, args.iter_max)
BATCH = 100
z = fsvae.sample_z(batch=BATCH)# (BATCH,10)

# digits = [int(i) for i in (input("Enter the digit(s) you want to generate: (insert more than 1 to generate mixed version of those digts :D):").split())]
digit = int(input("Enter your digit : "))
y = torch.zeros(10)
# for digit in digits:
y[digit] = 1
y = ut.duplicate(y,BATCH).reshape(BATCH,-1)#(BATCH,10)
x_mean = fsvae.compute_mean_given(z,y) #(BATCH,3072)
x_mean = torch.clip(x_mean,0,1)
# x_samples = ut.sample_gaussian(x_mean,0.1*torch.ones_like(x_mean))
np_x_samples = x_mean.detach().cpu().numpy().reshape(-1, 32, 32)

# Create a grid of 10x10
fig, axs = plt.subplots(10, 10, figsize=(9,9))
fig.subplots_adjust(hspace = 0, wspace = 0)
for ax in axs.ravel():
    ax.axis('off')
for i, ax in enumerate(axs.ravel()):
    ax.imshow(np_x_samples[i], cmap='gray',aspect='auto')
plt.savefig(f"visualize_BATCH_digit_fsvae_{digit}.png")
plt.show()