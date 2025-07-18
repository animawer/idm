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

def data_gen_dyadic(folder_path, tlen=30, gen_val=False, val_len=1000):
        print("load dataset")
        p_file = pathlib.Path(folder_path)
        db_g = {}
        db_k = {}
        shape_g = {}
        for p in p_file.glob("i*"): 
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
        bs_a = np.zeros(3)
        bs_b = np.zeros(3)
        for k in db_g.keys():
            feature_a = db_g[k][a_col].values.astype(np.float32)
            feature_b = db_g[k][b_col].values.astype(np.float32)
            for i in range(3):
                bs_a[i] += np.median(feature_a[:, :-1][:, i])
                bs_b[i] += np.median(feature_b[:, :-1][:, i])
        
        bs_a /= len(db_g.keys())
        bs_b /= len(db_g.keys())
        print ("angle bias:", bs_a, bs_b)
        for k in db_g.keys():
            feature_a = db_g[k][a_col].values.astype(np.float32)
            feature_b = db_g[k][b_col].values.astype(np.float32)
            for i in range(3):
                feature_a[:, :-1][:, i] = feature_a[:, :-1][:, i]-bs_a[i] + np.pi/4#np.median(feature_a[:, :-1][:, i])
                feature_b[:, :-1][:, i] = feature_b[:, :-1][:, i]-bs_b[i] + np.pi/4#np.median(feature_b[:, :-1][:, i])
            
            feature_a[:, :-1][:, 3:] /= 5
            feature_b[:, :-1][:, 3:] /= 5 
            print (feature_a.shape, feature_b.shape)
            if ("s001" in str(k)) and gen_val:
                twomin = 10*60*2
                print (k, twomin)
                feature_a_val = feature_a[len(feature_a)-twomin:]
                feature_b_val = feature_b[len(feature_b)-twomin:]
                feature_a = feature_a[:len(feature_a)-twomin]
                feature_b = feature_b[:len(feature_b)-twomin]
            
            tmp_d_left = []
            tmp_d_g = []
            for i in range(tlen*2, len(feature_a), 2):
                t_a = feature_a[i-tlen*2:i:2]
                t_b = feature_b[i-tlen*2:i:2]
                #print("t_right shape:", t_right.shape)
                t3 = np.concatenate([t_a.reshape(1, t_a.shape[0], t_a.shape[1]), 
                                        t_b.reshape(1, t_b.shape[0], t_b.shape[1])], 0)
                dataset.append(t3.reshape(1, t3.shape[0], t3.shape[1], t3.shape[2]))

            if ("s001" in str(k)) and gen_val:
                #print (feature_a_val)
                for i in range(tlen*2, len(feature_a_val), 2):
                    t_a = feature_a_val[i-tlen*2:i:2]
                    t_b = feature_b_val[i-tlen*2:i:2]
                    #print("t_right shape:", t_right.shape)
                    t3 = np.concatenate([t_a.reshape(1, t_a.shape[0], t_a.shape[1]), 
                                        t_b.reshape(1, t_b.shape[0], t_b.shape[1])], 0)
                    dataset_val.append(t3.reshape(1, t3.shape[0], t3.shape[1], t3.shape[2]))

                print (f"n_val_data:{len(dataset_val)}")
        dataset = np.concatenate(dataset, 0)
        if gen_val:
            #print (len(dataset_val))
            dataset_val = np.concatenate(dataset_val, 0)
        return dataset, dataset_val

def data_gen_kiq(folder_path, tlen=30, gen_val=False, val_len=1000):
    print("load dataset")
    p_file = pathlib.Path(folder_path)
    db_g = {}
    db_k = {}
    shape_g = {}
    for p in p_file.glob("Guest*"):
        nm = str(p.stem).split("_")
        s_p, lb = f"{nm[-4]}_{nm[-3]}_{nm[-2]}_{nm[-1]}", nm[-2]#str(p.name).split("_")[1], str(p.name).split("_")[4]
        db_g.update({s_p:pd.read_csv(p).fillna(method='ffill')})
        shape_g.update({s_p:db_g[s_p].fillna(method='ffill').values.shape[1]})
    for p in p_file.glob("KiQ*"):
        nm = str(p.stem).split("_")
        s_p, lb = f"{nm[-4]}_{nm[-3]}_{nm[-2]}_{nm[-1]}", nm[-2]#str(p.name).split("_")[1], str(p.name).split("_")[4]
        db_k.update({s_p:pd.read_csv(p).fillna(method='ffill')})
    #print (shape_g)
    max_key, max_val = max(shape_g.items(), key=lambda x: x[1])
    cols = db_g[max_key].columns
    del_c_idx = []
    #print (list(db_g.keys()))
    #print (np.sort(list(db_g.keys())))
    #exit(0)
    dataset = []
    dataset_val = []
    d1 = []
    d2 = []
    data_r = []
    data_l = []
    data_idx = []
    for k in np.sort(list(db_g.keys())):
        print (k)
        feature_g = db_g[k].values.astype(np.float32)
        feature_k = db_k[k].values.astype(np.float32)
        feature_g[:, :-1][:, 1::2] += 0.5
        feature_k[:, :-1][:, 1::2] += 0.5
        feature_g[:,:-1] *= 2
        feature_k[:,:-1] *= 2
        if ("session1" in str(k)) and gen_val:
            twomin = 10*60*2
            #print (twomin)
            feature_g_val = feature_g[len(feature_k)-twomin:]
            feature_k_val = feature_k[len(feature_k)-twomin:]
            feature_g = feature_g[:len(feature_g)-twomin]
            feature_k = feature_k[:len(feature_k)-twomin]
        #print (k, np.max(feature_g[:,0]), np.min(feature_g[:,0]),np.max(feature_g[:,1]), np.min(feature_g[:,1]) , np.max(feature_g[:,2]), np.min(feature_g[:,2]),np.max(feature_g[:,3]), np.min(feature_g[:,3]))
        #print (np.median(feature_g[feature_g[:,-1] > 0,-1]), np.median(feature_k[feature_k[:,-1] > 0,-1]))
        tmp_d_left = []
        tmp_d_g = []
        for i in range(tlen*2, len(feature_g), 2):
            t_g = feature_g[i-tlen*2:i:2]
            t_k = feature_k[i-tlen*2:i:2]
            #print("t_right shape:", t_right.shape)
            t3 = np.concatenate([t_g.reshape(1, t_g.shape[0], t_g.shape[1]), 
                                    t_k.reshape(1, t_k.shape[0], t_k.shape[1])], 0)
            dataset.append(t3.reshape(1, t3.shape[0], t3.shape[1], t3.shape[2]))
        if ("session1" in str(k)) and gen_val:
            #print (feature_g_val)
            for i in range(tlen*2, len(feature_g_val), 2):
                t_g = feature_g_val[i-tlen*2:i:2]
                t_k = feature_k_val[i-tlen*2:i:2]
                t3 = np.concatenate([t_g.reshape(1, t_g.shape[0], t_g.shape[1]), 
                                    t_k.reshape(1, t_k.shape[0], t_k.shape[1])], 0)
                dataset_val.append(t3.reshape(1, t3.shape[0], t3.shape[1], t3.shape[2]))
    del db_g, db_k
    dataset = np.concatenate(dataset, 0)
    if gen_val:
        #print (len(dataset_val))
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