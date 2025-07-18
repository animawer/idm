import numpy as np
import torch
import torch.functional as f
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import pathlib

def data_gen_dyadic_pose(folder_path, tlen=30, gen_val=False, val_len=1000):
        print("load dataset")
        p_file = pathlib.Path(folder_path)
        db_g = {}
        db_k = {}
        shape_g = {}
        for p in p_file.glob("i*"): #for p in p_file.glob("RIGHT*"):
            #print (p)
            nm = str(p.stem).split("_")
            s_p, lb = f"{nm[0]}", nm[0]#str(p.name).split("_")[1], str(p.name).split("_")[4]
            db_g.update({s_p:pd.read_csv(p).fillna(method='ffill')})
        print (shape_g)
        #max_key, max_val = max(shape_g.items(), key=lambda x: x[1])
        cols = db_g[list(db_g.keys())[0]].columns
        del_c_idx = []
        #print (list(db_g.keys()))
        #exit(0)
        dataset = []
        dataset_val = []
        d1 = []
        d2 = []
        data_r = []
        data_l = []
        data_idx = []
        a_col = [c for c in cols if "A_" in c]
        b_col = [c for c in cols if "B_" in c]
        print (a_col)
        print (b_col)
        db_keys = sorted(list(db_g.keys()))
        print (db_keys)
        for k in db_keys:
            feature_a = db_g[k][a_col].values.astype(np.float32)
            feature_b = db_g[k][b_col].values.astype(np.float32)
            feature_a[:, :-1][:, 1::2] += 0.5
            feature_b[:, :-1][:, 1::2] += 0.5 
            feature_a[:, :-1] *= 2.0
            feature_b[:, :-1] *= 2.0 

            tmp_d_left = []
            tmp_d_g = []
            tmp_a = feature_a[::2]
            tmp_b = feature_b[::2]
            for i in range(tlen, len(tmp_a)):
                t_a = tmp_a[i-tlen:i]
                t_b = tmp_b[i-tlen:i]
                t3 = np.concatenate([t_b.reshape(1, t_b.shape[0], t_b.shape[1]), 
                                        t_a.reshape(1, t_a.shape[0], t_a.shape[1])], 0)
                dataset.append(t3.reshape(1, t3.shape[0], t3.shape[1], t3.shape[2]))
        dataset = np.concatenate(dataset, 0)
        if gen_val:
            print (len(dataset_val))
            dataset_val = np.concatenate(dataset_val, 0)
        return dataset, dataset_val


class dataset(torch.utils.data.Dataset):
    def __init__(self, data, tlen=50, sep = 1):
        self.dataset = data
        self.tlen = tlen
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return x