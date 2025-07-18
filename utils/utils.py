import sys
import numpy as np
import torch
import torch.functional as f
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
sys.path.append("../")
from net.network import TransformerEnc
from utils.dataset import dataset
import pathlib
import matplotlib.pyplot as plt

#diffusion process 
def fw_diffusion(x, alpha_cum, nit):
    t = torch.randint(nit, (x.shape[0], 1)).squeeze()
    ext_alpha = torch.sqrt(alpha_cum[t]).view(-1, 1, 1, 1)
    ext_n_alpha = torch.sqrt(1.0 - alpha_cum[t]).view(-1, 1, 1, 1)
    eps = torch.randn(*x.shape)
    diff_x = ext_alpha * x + ext_n_alpha*eps
    return diff_x, t, eps

def fw_diffusion_at_t(x, alpha_cum, t=0):
    #t = torch.randint(nit, (x.shape[0], 1)).squeeze()
    ext_alpha = torch.sqrt(alpha_cum[t]).view(-1, 1, 1, 1).cuda(0)
    ext_n_alpha = torch.sqrt(1.0 - alpha_cum[t]).view(-1, 1, 1, 1).cuda(0)
    eps = torch.randn(*x.shape).cuda(0)
    diff_x = ext_alpha * x + ext_n_alpha*eps
    return diff_x, eps

# mask
def mask_generator(x):
    x_ratio = torch.clamp((torch.rand(x.shape[0])+0.8)/1.8, min=0.8, max=1.0)
    y_ratio = torch.clamp((torch.rand(x.shape[0])+0.8)/1.8, min=0.8, max=1.0) #0.75
    x_pos = torch.clamp(torch.rand(x.shape[0])*0.3, min=0.0, max=0.3)
    y_pos = torch.clamp(torch.rand(x.shape[0])*0.3, min=0.0, max=0.3)
    
    p_x = (x_pos * x.shape[2]).to(torch.long)
    p_y = (y_pos * x.shape[3]).to(torch.long)
    l_x = (x_ratio * x.shape[2]).to(torch.long)
    l_y = (y_ratio * x.shape[3]).to(torch.long)
    for i in range(len(x)):
        if p_x[i] + l_x[i] > x.shape[2]:
            l_x[i] = x.shape[2] - p_x[i]
        if p_y[i] + l_y[i] > x.shape[3]:
            l_y[i] = x.shape[3] - p_y[i]
    
    #print (p_x)
    #print (l_x)
    
    mask = torch.ones(x.shape)
    for i in range(len(mask)):
        rnd = np.random.rand()
        if rnd < 0.2:
            mask[i, :, p_x[i]:p_x[i]+l_x[i],p_y[i]:p_y[i]+l_y[i]] = 0
        elif rnd < 0.4:
            r =  np.random.rand()
            _y = 0.3 * torch.rand(1)+0.4 #0.4* torch.rand(1) +  0.3
            sel_idx = int(_y*(mask.shape[-1]-1))
            perm = torch.randperm(mask.shape[-1])
            perm = perm[perm>0]
            idx = perm[:sel_idx]
            mask[i, :, :, idx] = 0

        
        else:
            _y = 0.3 * torch.rand(1)+0.4
            _p_y = (_y * x.shape[2]).to(torch.long)#torch.clamp((torch.rand(x.shape[0])+0.25)/1.25, min=0.75, max=1.0)
            if np.random.rand() > 0.5:
                mask[i, :, _p_y:, :] = 0
                #mask[i, :, :,p_y[i]:p_y[i]+l_y[i]] = 0
            else:
                _i = np.random.choice([0,1])
                mask[i, _i, _p_y:, :] = 0
    return mask


def plt_mediapipe_pose(pose_vector, ax=None, size=None, save=None, color="black"):
    connect_mat = {0:[1, 4],
                   1:[0, 2],
                   2:[1, 3],
                   3:[2, 7],
                   4:[0, 5],
                   5:[4, 6],
                   6:[5, 8],
                   7:[3],
                   8:[6],
                   9:[10],
                   10:[9],
                   11:[12, 13, 23],
                   12:[11, 14, 24],
                   13:[11, 15],
                   14:[12, 16],
                   15:[13, 21, 17, 19],
                   16:[14, 18, 20, 22],
                   17:[15, 19],
                   18:[16, 20],
                   19:[15, 17],
                   20:[16, 18],
                   21:[15],
                   22:[16],
                   23:[11, 24, 25],
                   24:[12, 23, 26],
                   25:[23, 27],
                   26:[24, 28],
                   27:[25, 29, 31],
                   28:[26, 30, 32],
                   29:[27, 31],
                   30:[28, 32],
                   31:[27, 29],
                   32:[28, 30]
                  }
    # 0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24
    _id = {0:0, 2:1, 5:2, 9:3, 10:4, 11:5, 12:6, 13:7, 14:8, 15:9, 16:10, 19:11, 20:12, 23:13, 24:14}
    connect_mat2 = {0:[],
                    _id[2]:[],
                    _id[5]:[],
                    _id[9]:[_id[10]],
                    _id[10]:[_id[9]],
                   _id[11]:[_id[12], _id[13], _id[23]],
                   _id[12]:[_id[11], _id[14], _id[24]],
                   _id[13]:[_id[11], _id[15]],
                   _id[14]:[_id[12], _id[16]],
                   _id[15]:[_id[13], _id[19]],
                   _id[16]:[_id[14], _id[20]],
                   _id[19]:[_id[15]],
                   _id[20]:[_id[16]],
                   _id[23]:[_id[11], _id[24]],
                   _id[24]:[_id[12], _id[23]],
                  }
    
    # 0, 11, 12, 13, 14, 15, 16, 23, 24
    _id = {0:0, 11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 19:7, 20:8, 23:9, 24:10}
    connect_mat3 = {0:[],
                   _id[11]:[_id[12], _id[13], _id[23]],
                   _id[12]:[_id[11], _id[14], _id[24]],
                   _id[13]:[_id[11], _id[15]],
                   _id[14]:[_id[12], _id[16]],
                   _id[15]:[_id[13], _id[19]],
                   _id[16]:[_id[14], _id[20]],
                   _id[19]:[_id[15]],
                   _id[20]:[_id[16]],
                   _id[23]:[_id[11], _id[24]],
                   _id[24]:[_id[12], _id[23]],
                  }
    if size is None:
        x_w = 2160
        y_w = 3840
    else:
        x_w = size[0]
        y_w = size[1]
    if ax is None:
        fig = plt.figure(figsize=(9, 12))
        ax = fig.add_subplot(111)
    #ax.cla()
    if len(pose_vector) == 15*2+1:
        pose_vector = np.concatenate([pose_vector[:2], pose_vector[10:]], axis=-1)
    for index, i in enumerate(range(0, len(pose_vector)-len(pose_vector)%2, 2)):
        #print (index, i)
       
        ax.scatter(x_w*pose_vector[i], -y_w*pose_vector[i+1], color=color)
        if len(pose_vector) == 33*2:
            m = connect_mat[index]
        elif len(pose_vector) == 15*2+1:
            m = connect_mat2[index]
        else:
            m = connect_mat3[index]
        for k in m:
            ax.plot(x_w*pose_vector[[i, k*2]], -y_w*pose_vector[[i+1, k*2+1]], color="gray")
    ax.vlines(-0.8*2, -1*2, -1*2+0.5*pose_vector[-1],linewidth=4.0) #音
    #if save is not None:   
    #plt.xlim(0, x_w)
    #plt.ylim(0, y_w-1000)
    #return im1+im2
    #if show:
    #    plt.show()

from scipy import linalg
class EmbeddingSpaceEvaluator:
    def __init__(self, real_feat_list, gen_feat_list):

        # storage
        #self.context_feat_list = []
        self.real_feat_list = real_feat_list#[]
        self.generated_feat_list = gen_feat_list#[]
        #self.recon_err_diff = []

    
    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)