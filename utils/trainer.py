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
from utils.dataset import data_gen_kiq, dataset
from utils.utils import fw_diffusion, mask_generator
import pathlib

def trainer(profile, data_loader, cuda=0, optimizer=None, epoch=100, save_path="./model/", criterion=None, pos=True):
    nit=profile["nit"]
    trans_model = TransformerEnc(hidden=profile["hidden"], nhead=profile["nhead"], num_layer = profile["num_layer"], shape=profile["shape"], nit=profile["nit"])
    if cuda is not None:
        trans_model = trans_model.cuda(cuda)
    #optimizer = optim.RMSprop(trans_model.parameters(), lr = 5e-4)
    beta = torch.linspace(1e-4, 0.02, nit) 
    alpha = 1.0-beta
    alpha_cum = torch.tensor([alpha[0]] + [torch.prod(alpha[:i+1]) for i in range(1, len(beta))])

    if optimizer is None:
        optimizer = optim.AdamW(trans_model.parameters(), lr = 1e-4, betas=(0.9, 0.999))#, amsgrad=True)#, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=4*len(data_loader), mode="triangular2", cycle_momentum=False)

    epoch = epoch
    min_loss = 1e+5
    cntr = 0
    losses = []
    c_f = False
    if criterion is None:
        criterion = F.l1_loss
        c_f = True
    for e in range(epoch):
        ep_loss = 0
        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            if cuda is not None:
                diff_x, t, eps = fw_diffusion(data.squeeze(), alpha_cum, nit=len(alpha_cum))
                o_data = data.cuda(cuda)
                mask = torch.zeros(o_data.shape).cuda(cuda)
                mask = mask_generator(o_data).cuda(cuda)
                diff_x = diff_x.cuda(0)
                t = t.view(-1, 1).cuda(0)
                eps = eps.cuda(0)
                diff_x = o_data*mask + (diff_x*(1-mask))
                eps = o_data*mask + (eps*(1-mask))
            result = trans_model(diff_x, t, pos=pos)
            result = o_data*mask + (result*(1-mask))
            #result_p = result[:,:,:,:-1]
            #result_v = result[:,:,:,-1]
            delta_rec = result[:,:,1:,:-1]-result[:,:,:-1,:-1]
            delta_true = o_data[:,:,1:,:-1]-o_data[:,:,:-1,:-1]
            loss =  criterion(result, o_data)
            if c_f == True: 
                loss += 0.25*criterion(delta_rec, delta_true)
            loss.backward()
            optimizer.step()
            ep_loss += loss.detach().cpu().data
            print ("\rbatches: {}".format(idx), end="")
            scheduler.step()
        losses += [ep_loss.numpy()/idx]
        print ("  epoch: {}, loss: {:.3f}".format(e, ep_loss/idx))
        model_path = pathlib.Path(save_path)
        name = model_path / f"latest_ddim.pth"
        torch.save(trans_model.state_dict(), name)
        if min_loss > ep_loss/idx:
            name = model_path / f"minimum_ddim.pth"
            torch.save(trans_model.state_dict(), name)
            min_loss = ep_loss/idx
            cntr = 0
        else:
            cntr += 1
        if cntr > 9:
            break
    return trans_model, losses


if __name__ == '__main__':
    tlen = 50
    profile = {
        "hidden":512,
        "nhead":8,
        "num_layer":4,
        "shape":(tlen,23),
        "nit":200
    }
    nit = profile["nit"]
    beta = torch.linspace(1e-4, 0.02, nit) 
    #beta = _beta[::1000//nit]
    alpha = 1.0-beta
    alpha_cum = torch.tensor([alpha[0]] + [torch.prod(alpha[:i+1]) for i in range(1, len(beta))])
    
    base_path = pathlib.Path("./for_learning/")
    tr, te= data_gen_kiq(base_path, tlen = tlen, gen_val=True)
    _train = torch.from_numpy(np.asarray(tr)).to(torch.float)
    trainset = dataset(_train)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    _test = torch.from_numpy(np.asarray(te)).to(torch.float)
    testset = dataset(_test)
    data_loader_te = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
    
    cuda = 0
    hidden = 512
    trans_model = TransformerEnc(hidden=profile["hidden"], nhead=profile["nhead"], num_layer = profile["num_layer"], shape=profile["shape"], nit=profile["nit"])
    if cuda is not None:
        trans_model = trans_model.cuda(cuda)
    trainer(trans_model, nit=nit, alpha_cum=alpha_cum, cuda=cuda, optimizer=None, epoch=100, save_path="./model/", criterion=None)