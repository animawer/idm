import numpy as np
import torch
import torch.functional as f
import torch.nn as nn
import torch.optim as optim
import math
import pathlib

class Conversion(nn.Module):
    def __init__(self, output_shape):
        super(Conversion, self).__init__()
        self.output_shape = output_shape
    def forward(self, x):
        return x.view(len(x), self.output_shape[0], self.output_shape[1], self.output_shape[2])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].reshape(1, x.size(1), self.pe.shape[-1])
        return x

class TransformerEnc(nn.Module):
    def __init__(self, hidden, nhead=8, num_layer = 6, shape=(50, 22), nit=300):
        super(TransformerEnc, self).__init__()
        self.shape = shape

        self.in_linear = nn.Sequential(
            nn.Linear(shape[1], hidden//2),
        )
        self.emb = nn.Embedding(nit, hidden)
        self.pe = PositionalEncoding(hidden, max_len=shape[0]+1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, dim_feedforward=hidden, nhead=nhead, dropout=0.2, batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.out_linear1 = nn.Sequential(
            nn.Linear(hidden//2, shape[1]),
        )
        self.out_linear2 = nn.Sequential(
            nn.Linear(hidden//2, shape[1]),
        )
        
    def forward(self, x, t, ):
        #if lab is not None:
        lab = self.emb(t)
        xshape = x.shape
        x = self.in_linear(x)
        x = torch.cat([x[:,0],x[:,1]], -1)
        x = torch.cat([lab, x], 1)
        x = self.pe(x)
        x = self.transformer_encoder(x)[:,1:]
        #print (x.shape)
        x1 = self.out_linear1(x[:,:,:x.shape[-1]//2].reshape(xshape[0], 1, x.shape[1], x.shape[2]//2))
        x2 = self.out_linear2(x[:,:,x.shape[-1]//2:].reshape(xshape[0], 1, x.shape[1], x.shape[2]//2))
        x = torch.cat([x1, x2], 1)
        return x
    

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net


class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
        )

        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
            #nn.Dropout(0.5),
            #nn.Linear(384, 256),  # for 34 frames
            #nn.Linear(640, 256), # for 50
            nn.Linear(224, 128), # for25
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64),
            #nn.Linear(256, 128),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            #nn.Linear(128, 32),
        )

        self.fc_mu = nn.Linear(64, 32)
        #self.fc_logvar = nn.Linear(32, 32)

    def forward(self, poses, variational_encoding):
        # encode
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        #print (poses.shape)
        out = self.net(poses)
        out = out.flatten(1)
        #print (out.shape)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        #logvar = self.fc_logvar(out)

        #if variational_encoding:
        #    z = reparameterize(mu, logvar)
        #else:
        #    z = mu
        return mu #z, mu, logvar
    
class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False):
        super().__init__()

        feat_size = 32
        self.pre_net = nn.Sequential(
            nn.Linear(feat_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, length*4),
        )
        #else:
        #    assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        #print (out.shape)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, n_frames, pose_dim):#
        super().__init__()
        self.context_encoder = None
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)
        self.mode = "pose"

    def forward(self, poses, input_mode=None, variational_encoding=False):
        if input_mode is None:
            assert self.mode is not None
            input_mode = self.mode

        pose_feat = self.pose_encoder(poses, variational_encoding)
        out_poses = self.decoder(pose_feat)#, pre_poses)

        return pose_feat, out_poses


if __name__ == '__main__':
    enc = TransformerEnc(hidden=128, nhead=8, num_layer = 6, shape=(28, 32), nit=100)
    print (enc)