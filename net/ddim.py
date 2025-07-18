import numpy as np
import sys
import torch
import torch.functional as f
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
sys.path.append("../")
from net.network import TransformerEnc
from utils.utils import fw_diffusion, plt_mediapipe_pose
from utils.dataset import data_gen_kiq, dataset
import pathlib
import matplotlib.pyplot as plt

def ddpm_forward(trans_model, data, mask, alpha, alpha_cum, nit, cuda=0):
    noise = torch.randn(data.shape)
    noise = noise.cuda(cuda)
    xt = data*mask + (noise*(1-mask))
    cand = list(range(nit))[::nit//10][::-1]
    time = torch.tensor([0]*len(xt)).view(-1,1).cuda(0)
    with torch.no_grad():
        for idx, _d in enumerate(cand):
            z = 0
            var = (1-alpha[_d])/(torch.sqrt(1-alpha_cum[_d]))
            time[:,0] = _d
            gap = nit//100
            if _d > 0:
                _nd = cand[idx+1] 
                xzero =  trans_model(xt, time)
                xt = torch.sqrt(alpha_cum[_nd])*xzero + torch.sqrt(1-alpha_cum[_nd])*((xt-torch.sqrt(alpha_cum[_d])*xzero)/torch.sqrt(1-alpha_cum[_d]))
            else:
                xt = trans_model(xt, time)
            xt = data*mask + (xt*(1-mask))
    return xt.detach()

#profile: hidden, nhead=8, num_layer = 6, shape=(50, 22), nit=300
class ddim_generator(nn.Module):
    def __init__(self, profile, model_path=None, _cuda=None, mask_mode="forecast"):
        super(ddim_generator, self).__init__()
        d_shape = list(profile["shape"])
        self._cuda = _cuda
        self.nit = profile["nit"]
        tlen = profile["tlen"]
        self.beta = torch.linspace(1e-4, 0.025, self.nit) 
        #beta = _beta[::1000//nit]
        self.alpha = 1.0-self.beta
        self.alpha_cum = torch.tensor([self.alpha[0]] + [torch.prod(self.alpha[:i+1]) for i in range(1, len(self.beta))])
        
        self.trans_model = TransformerEnc(profile["hidden"], nhead=profile["nhead"], num_layer=profile["num_layer"], shape=profile["shape"], nit=self.nit)
        if model_path is not None:
            self.trans_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print ("load model: success")
        if _cuda is not None:
            self.trans_model.cuda(_cuda)
        #if not train:
        self.trans_model.eval()
        self.mask_mode = mask_mode
        if mask_mode == "forecast":
            mask_shape = [1, 2] + d_shape
            self.mask = torch.ones(mask_shape)
            self.mask[:,:,tlen//2:,:] = 0
        elif mask_mode == "interpolation":
            mask_shape = [1, 2] + d_shape
            self.mask = torch.ones(mask_shape)
            self.mask[:,:,tlen//2:-2,:] = 0
        else:
            self.mask = mask_mode
        if _cuda is not None:
            self.mask = self.mask.cuda(_cuda)
        assert type(self.mask) != str, 'mask is not correctly defined'
        if self._cuda is not None:
            self.forward(torch.randn((16,2,*d_shape)).cuda(_cuda))
            self.forward(torch.randn((16,2,*d_shape)).cuda(_cuda))
        else:
            self.forward(torch.randn((16,2,*d_shape)))
            self.forward(torch.randn((16,2,*d_shape)))

    def forward(self, data, ncal=20, mask_out=False):
        #if type(self.mask_model) == str: 
        if len(data) == 1:
            M = self.mask
        else:
            M = torch.zeros(data.shape)
            if self._cuda is not None:
                M = M.cuda(self._cuda)
            M[:] = self.mask
        noise = torch.randn(data.shape)
        time = torch.tensor([0]*len(data)).view(-1,1) #バッチサイズ分のddim時刻情報
        if self._cuda is not None:
            noise = noise.cuda(self._cuda)
            time  = time.cuda(self._cuda)
        xt = data*M + (noise*(1-M))
        cand = list(range(self.nit))[::self.nit//ncal][::-1]
        with torch.no_grad():
            for idx, _d in enumerate(cand):
                z = 0
                time[:,0] = _d
                if _d > 0:
                    _nd = cand[idx+1] 
                    xzero =  self.trans_model(xt, time)
                    xt = torch.sqrt(self.alpha_cum[_nd])*xzero + torch.sqrt(1-self.alpha_cum[_nd])*((xt-torch.sqrt(self.alpha_cum[_d])*xzero)/torch.sqrt(1-self.alpha_cum[_d]))
                else:
                    xt = self.trans_model(xt, time)
                xt = data*M + (xt*(1-M))
        x_out = xt.detach()
        #x_out[:,:,:,:3] -= np.pi/4
        if mask_out:
            return x_out, M
        else:
            return x_out

    #def forward(self, x):



if __name__ == '__main__':
    tlen=50
    profile = {
        "hidden":512,
        "nhead":4,
        "num_layer":4,
        "shape":(tlen,23),
        "tlen":tlen
    }
    base_path = pathlib.Path("./for_learning/")
    tr, te= data_gen_kiq(base_path, tlen = tlen, gen_val=True)

    _test = torch.from_numpy(np.asarray(te)).to(torch.float)
    testset = dataset(_test)
    data_loader_te = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    model_path = "./model/minimum_ddim.pth"
    motion_generator = ddim_generator(profile, model_path=model_path, _cuda=0, mask_mode="forecast")

    # Just visualize
    import copy, time
    result_all = []
    data_all = []
    cuda = 0
    for idx, data in enumerate(data_loader_te):
        _o_data = copy.deepcopy(data).cuda(cuda)
        o_data = data.cuda(cuda)
        res = []
        start = time.time()
        with torch.no_grad():
            res, mask = motion_generator(o_data)
        print ("est. time:", time.time()-start)
        data_all += (data/2).numpy().tolist()
        rec_data = _o_data*mask + res * (1-mask)
        rec_data[:,:,:,-1] = _o_data[:,:,:,-1]
        result_all += (rec_data/2).detach().cpu().numpy().tolist()
    print ("dump images...")
    data_all = np.array(data_all)
    result_all = np.array(result_all)
    def update(i, tlen, d_r, d_t, d_p, ax):
        x_w = 1
        y_w = 1
        ax[0].cla()
        ax[0].set_xlim(-x_w*2, x_w*2)
        ax[0].set_ylim(-y_w*2, y_w*2)
        ax[1].cla()
        ax[1].set_xlim(-x_w*2, x_w*2)
        ax[1].set_ylim(-y_w*2, y_w*2)
        ax[2].cla()
        ax[2].set_xlim(-x_w*2, x_w*2)
        ax[2].set_ylim(-y_w*2, y_w*2)
        pose_vector1 = d_r[i]
        pose_vector2 = d_t[i]
        pose_vector3 = d_p[i]
        if i >= tlen//2:
            color="red"
        else:
            color="black"
        plt_mediapipe_pose(pose_vector1, ax[0], size=[x_w, y_w], color=color)
        plt_mediapipe_pose(pose_vector2, ax[1], size=[x_w, y_w], color="black")
        plt_mediapipe_pose(pose_vector3, ax[2], size=[x_w, y_w], color="blue")
        fig.savefig(f"./img/ddim_img_{i}.png")
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(12,6))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    fps = 5
    est = False
    mov_idx = 0
    d_r = result_all[mov_idx, 1, :, :]*2
    d_t = data_all[mov_idx, 1, :, :]*2
    d_p = data_all[mov_idx, 0, :, :]*2
    # dump images
    for i in range(len(d_r)):
        update(i, tlen, d_r, d_t, d_p, ax)