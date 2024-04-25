import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m,v = self.enc.encode(x)
        z = ut.sample_gaussian(m,v)
        out = self.dec.decode(z)

        kl = ut.kl_normal(m,v,self.z_prior_m,self.z_prior_v)
        rec = -ut.log_bernoulli_with_logits(x,out)
        nelbo = kl + rec

        rec = torch.mean(rec)
        kl = torch.mean(kl)
        nelbo = torch.mean(nelbo)
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations (100,784)
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        m,v = self.enc.encode(x)
        m = ut.duplicate(m,iw)
        v = ut.duplicate(v,iw)
        x = ut.duplicate(x,iw)
        z = ut.sample_gaussian(m,v)
        out = self.dec.decode(z)
        kl = ut.log_normal(z,m,v) - ut.log_normal(z,self.z_prior[0],self.z_prior[1])
        rec = -ut.log_bernoulli_with_logits(x,out)
        nelbo = kl+rec
        # elbo có dạng log(p(...))=> cần exp lên để thành p(...) rồi lấy mean(p(...)) và lấy log(mean(...))
        # nhớ đổi dấu nelbo do cần tính elbo, niwae đổi dấu do cần tính min của số âm <=> max của số dương
        niwae = -ut.log_mean_exp(-nelbo.reshape(iw,-1) , dim=0)
        niwae , kl , rec = niwae.mean(),kl.mean(),rec.mean()

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),# (batch,z_dim)
            self.z_prior[1].expand(batch, self.z_dim)) #(batch,z_dim)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
