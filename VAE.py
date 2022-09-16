from __future__ import absolute_import

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

device = 'CUDA'

class Encoder(nn.Module):
    def __init__(self, feature_size, latent_size=10, condition_dim=2):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.condition_dim = condition_dim
        # encode
        self.fc1 = nn.Linear(feature_size + self.condition_dim, feature_size)
        self.fc1_2 = nn.Linear(feature_size, latent_size)
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.fc3 = nn.Linear(latent_size, latent_size)

        self.norm = nn.BatchNorm1d(feature_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.norm(xc)
        xc = self.fc1_2(xc)
        hidden = torch.tanh(xc)
        mu = self.fc2(hidden)
        logvar = self.fc3(hidden)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, feature_size, latent_size=10, condition_dim=2, decode_bn_affine=True):
        super(Decoder, self).__init__()
        self.feature_size = feature_size
        self.condition_dim = condition_dim
        # decode
        self.fc1 = nn.Linear(latent_size + condition_dim, feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size * 2)
        self.fc3 = nn.Linear(feature_size * 2, feature_size)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm1d(feature_size)
        self.norm2 = nn.BatchNorm1d(feature_size * 2)
        print('2 BN at decoder output')
        self.norm3_sou = nn.BatchNorm1d(feature_size, affine=decode_bn_affine)
        self.norm3_tar = nn.BatchNorm1d(feature_size, affine=decode_bn_affine)

    def kl_2_norm(self, mu1, var1, mu2, var2):
        loss = torch.log(torch.sqrt(var2 / var1)) + (var1 + torch.square(mu1 - mu2)) / 2 / var2 - 0.5
        loss = torch.sum(loss)
        return loss

    def decode_forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        hidden = self.fc1(xc)
        hidden = self.relu(hidden)
        hidden = self.norm1(hidden)
        hidden = self.fc2(hidden)
        hidden = self.relu(hidden)
        hidden = self.norm2(hidden)
        out = self.fc3(hidden)
        return out

    def transfer_forward(self, x, c):
        out = self.decode_forward(x, c)
        out_mean, out_var = torch.mean(out, dim=0), torch.var(out, dim=0)
        sou_mean, sou_var = self.norm3_sou.running_mean, self.norm3_sou.running_var
        kl_loss = self.kl_2_norm(out_mean, out_var, sou_mean.detach(), sou_var.detach())
        out = self.norm3_sou(out)
        return out, kl_loss

    def forward(self, x, c):
        out = self.decode_forward(x, c)
        out_target, out_source = torch.chunk(out, 2, dim=0)
        recon_tar = self.norm3_tar(out_target)
        recon_sou = self.norm3_sou(out_source)
        out = torch.cat([recon_tar, recon_sou], dim=0)
        return out


class VAE(nn.Module):
    def __init__(self, feature_size, latent_size=128, condition_dim=2, KL_loss2=1,
                 feat_vox_train_mean=None, feat_vox_train_std=None,
                 feat_cn_train_mean=None, feat_cn_train_std=None,
                 rate1=1.0, rate2=0.01, rate3=0.1, rate4=0.1, cos_thres=1):
        super(VAE, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.condition_dim = condition_dim
        self.KL_loss2 = KL_loss2

        self.loss_reduction = 'sum'
        self.loss_reduction_fc = lambda x: torch.sum(x)

        self.feat_vox_train_mean = feat_vox_train_mean  # please calculate it on your trained model and dataset
        self.feat_vox_train_std = feat_vox_train_std    # please calculate it on your trained model and dataset
        self.feat_cn_train_mean = feat_cn_train_mean    # please calculate it on your trained model and dataset
        self.feat_cn_train_std = feat_cn_train_std      # please calculate it on your trained model and dataset

        self.label1 = torch.tensor([[1, 0]]).float().cuda()  # label for target domain data
        self.label2 = torch.tensor([[0, 1]]).float().cuda()  # label for source domain data

        # pre norm
        print('No BN on vae input feat')
        decode_bn_affine = True
        # encode
        self.encoder = Encoder(feature_size, latent_size, condition_dim=condition_dim)
        # decode
        self.decoder = Decoder(feature_size, latent_size, condition_dim=condition_dim,
                               decode_bn_affine=decode_bn_affine)

        self.prior = nn.Linear(self.condition_dim, self.latent_size)

        self.recon_loss = self.reconstruction_loss_L2
        self.KL_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.rate1 = rate1  # loss weight for reconstruction
        self.rate2 = rate2  # loss weight for KL-div
        self.rate3 = rate3  # loss self-supervised cosine1
        self.rate4 = rate4  # loss self-supervised cosine2
        self.cos_thres = cos_thres

    def reconstruction_loss_L2(self, x, recon_x):
        loss = x - recon_x
        loss = self.loss_reduction_fc(loss * loss)
        return loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def evaluate_forward(self, x):  # For evaluation
        p_num = x.size(0)
        if self.feat_cn_train_mean is not None:
            x = (x - self.feat_cn_train_mean) / self.feat_cn_train_std
        out_feat, _ = self.vae_model.encoder(x, self.label1.repeat([p_num, 1]))
        target_mean = self.prior(self.label1)
        source_mean = self.prior(self.label2)
        out_feat = out_feat - target_mean + source_mean
        out_feat, _ = self.decoder.transfer_forward(out_feat, self.label2.repeat([p_num, 1]))
        return out_feat

    def forward(self, x_target, c_target, x_source, c_source):  # For training
        # x: noisy embeddings
        # c: condition label

        # Normalize the embeddings
        if self.feat_vox_train_mean is not None:
            x_target = (x_target - self.feat_cn_train_mean) / self.feat_cn_train_std
            x_source = (x_source - self.feat_vox_train_mean) / self.feat_vox_train_std

        x = torch.cat([x_target, x_source], dim=0)
        c = torch.cat([c_target, c_source], dim=0)

        input_size = x.size()
        batch_size_all = input_size[0]
        batch_size_tar = batch_size_all // 2

        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c)

        rec_loss = self.recon_loss(x, recon_x)
        x_prior = self.prior(c)
        kl_loss = self.KL_loss(mu - x_prior, logvar)
        #########
        x_prior_target, x_prior_source = torch.chunk(x_prior, 2, dim=0)
        z_target = z[:batch_size_tar, :]

        # Transfer the embeddings on the latent space
        z_target_source = z_target - x_prior_target + x_prior_source

        recon_x_target_source, kl_loss2 = self.decoder.transfer_forward(z_target_source, c_source)
        # kl_loss2: Minimize the KL-div between the transferred embeddings and the source's distribution.

        rec_loss /= batch_size_all
        kl_loss /= batch_size_all
        kl_loss = kl_loss + self.KL_loss2 * kl_loss2

        # Calculate the self-supervised cosine loss
        cos_latent = F.cosine_similarity(recon_x_target_source.unsqueeze(1), x_source.unsqueeze(0), dim=2)
        cos_latent = (cos_latent + self.cos_thres) / (1 + self.cos_thres)
        cos_latent = torch.relu(-torch.log(1.0 - cos_latent + 1e-5))
        cos_loss1 = torch.sum(cos_latent) / batch_size_tar / batch_size_tar

        cos_latent = F.cosine_similarity(recon_x_target_source.unsqueeze(1), recon_x_target_source.unsqueeze(0), dim=2)
        cos_latent = (cos_latent + self.cos_thres) / (1 + self.cos_thres)
        cos_latent = torch.relu(-torch.log(1.0 - cos_latent + 1e-5))
        cos_mask = torch.eye(batch_size_tar).cuda()
        cos_latent = (1.0 - cos_mask) * cos_latent
        cos_loss2 = torch.sum(cos_latent) / batch_size_tar / (batch_size_tar - 1)

        cos_loss1 = cos_loss1 * self.feature_size
        cos_loss2 = cos_loss2 * self.feature_size

        cos_loss = self.rate3 * cos_loss1 + self.rate4 * cos_loss2

        loss = self.rate1 * rec_loss + self.rate2 * kl_loss + cos_loss

        return recon_x, recon_x_target_source, loss


if __name__ == '__main__':
    model = VAE(256, 10).cuda()

    x = torch.ones([10, 256]).cuda()
    y = torch.ones([10, 2]).cuda()

    out = model(x, y, x, y)
    print(out[-1])
