import numpy as np
import os
import pandas as pd

def q_validities(pair_df):
    cue_valies=[]
    for col in range(1,pair_df.shape[1]):
        #num_zeros = np.sum(pair_df.iloc[:,col]==0)
        pair_df_xitmp = pair_df.iloc[:,col]
        pair_df_ytmp = pair_df.iloc[:,0]
        pair_df_ytmp = pair_df_ytmp[pair_df_xitmp!=0]
        pair_df_xitmp = pair_df_xitmp[pair_df_xitmp!=0]
        cue_vali = np.sum(pair_df_ytmp==pair_df_xitmp)/float(len(pair_df_ytmp))
        cue_valies.append(cue_vali)
    return np.array(cue_valies)

def paired_data(ds_i, flip_direction=0):
    tmp0 = pd.read_csv(data_dir + ds_fn_list[ds_i], sep="\s+|\t+")
    tmp = tmp0.reindex(np.random.permutation(tmp0.index))
    #tmp = tmp.drop(range(len(tmp)-5))
    y_tmp = tmp.iloc[:,2]
    x_tmp = tmp.iloc[:,4:]
    num_pairs = len(y_tmp)*(len(y_tmp)-1) #/2
    ph_array = np.zeros((num_pairs,x_tmp.shape[1]+1))
    ph_array.fill(np.nan)
    pair_df_header = np.hstack([y_tmp.name,x_tmp.columns.values])
    pair_df = pd.DataFrame(ph_array,columns=pair_df_header)
    i = 0
    n = 0
    for y_i in y_tmp:
        num_y_j = len(y_tmp) - (i + 1)
        for j in range(i+1,i+1+num_y_j):
            pair_y = (y_i - y_tmp.iloc[j])/np.abs(y_i - y_tmp.iloc[j])
            pair_x = x_tmp.iloc[i,:] - x_tmp.iloc[j,:]
            pair_df.iloc[n,0] = pair_y
            pair_df.iloc[n,1:] = pair_x
            n += 1
            #let's switch directions now
            pair_y = (y_tmp.iloc[j] - y_i)/np.abs(y_tmp.iloc[j] - y_i)
            pair_x = x_tmp.iloc[j,:] - x_tmp.iloc[i,:]
            pair_df.iloc[n,0] = pair_y
            pair_df.iloc[n,1:] = pair_x
            n += 1
        i += 1
    if flip_direction==1:
        chng_q_valis = q_validities(pair_df)<0.5
        for col in range(1,pair_df.shape[1]):
            if chng_q_valis[col-1]:
                pair_df.iloc[:,col] = pair_df.iloc[:,col]*-1
    qvalis = q_validities(pair_df)
    ttb_inds = np.argsort(qvalis)+1
    ttb_inds = ttb_inds[::-1]
    return pair_df, ttb_inds, qvalis

np.random.seed(101)
parent_dir = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/'
data_dir = parent_dir + 'data/'
tmp_pair_data_dir = parent_dir + 'tmp_pair_data/'
ds_fn_list = os.listdir(data_dir)

all_ttb_inds=[]
all_qvalis=[]
for ds_i in range(len(ds_fn_list)):
    pair_df, ttb_inds, qvalis = paired_data(ds_i)
    print(qvalis)
    print(pair_df.shape)
    #print(pair_df.iloc[:,0].mean())
    #exit()
    pair_df.to_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_i], index=False)
    all_ttb_inds.append(ttb_inds)
    all_qvalis.append(qvalis)

np.save(tmp_pair_data_dir + 'ttb_inds',all_ttb_inds)
np.save(tmp_pair_data_dir + 'qvalis',all_qvalis)
