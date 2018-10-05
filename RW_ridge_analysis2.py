import numpy as np
import os
import pandas as pd
import random

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

def compute_qvalis(pair_df):
    #assumes first column of pair_df is Y values
    chng_q_valis = q_validities(pair_df)<0.5
    flippers=[]
    for col in range(1,pair_df.shape[1]):
        flipper=0
        if chng_q_valis[col-1]:
            pair_df.iloc[:,col] = pair_df.iloc[:,col]*-1
            flipper=1
        flippers.append(flipper)
    qvalis = q_validities(pair_df)
    ttb_inds = np.argsort(qvalis)+1
    ttb_inds = ttb_inds[::-1]
    return pair_df, ttb_inds, qvalis, flippers

def paired_data(ds_i):
    tmp0 = pd.read_csv(data_dir + ds_fn_list[ds_i], sep="\s+|\t+")
    tmp = tmp0.reindex(np.random.permutation(tmp0.index))
    #tmp = tmp.drop(range(len(tmp)-5))
    y_tmp = tmp.iloc[:,2]
    x_tmp = tmp.iloc[:,4:]
    num_pairs = len(y_tmp)*(len(y_tmp)-1)/2
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
        i += 1
    pair_df, ttb_inds, qvalis, _ = compute_qvalis(pair_df)
    return pair_df, ttb_inds, qvalis

def tallying(tmp0):
    tally_up = np.sum(tmp0.iloc[:,1:],axis=1)
    for i in range(len(tally_up)):
        if tally_up[i]==0:
            coin_flip = random.choice([1,-1])
            tally_up[i] = coin_flip
    tally_preds = tally_up/np.abs(tally_up)
    return tally_preds

def ttbing(ttb_inds, tmp0):
    ttb_preds = np.zeros((tmp0.shape[0],))
    for i in range(len(ttb_inds)):
        x_i = tmp0.iloc[:,ttb_inds[i]]
        msk = ttb_preds!=0
        x_i[msk] = 0
        ttb_preds += x_i
    for j in range(len(ttb_preds)):
        if ttb_preds[j] == 0:
            coin_flip = random.choice([1,-1])
            ttb_preds[j] = coin_flip
        return ttb_preds

def compute_ttb_prior(ttb_inds):
    ttb_prior = np.zeros((len(ttb_inds),))
    reverse_ttb_inds = ttb_inds[::-1] - 1 #order from worst to best cue & 0-based
    for n in range(len(ttb_inds)):
        prior[reverse_ttb_inds[n]] = 2**n
    return ttb_prior

np.random.seed(101)
parent_dir = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/'
data_dir = parent_dir + 'data/'
tmp_pair_data_dir = parent_dir + 'tmp_pair_data/'
ds_fn_list = os.listdir(data_dir)
all_ttb_inds = np.load(tmp_pair_data_dir + 'ttb_inds.npy') #in same order as data_dir when loaded through listdir

'''
paired_ds_list = ['pair_attractiveness.men.txt', 'pair_dropout.txt', 'pair_cloud.txt',
 'pair_bodyfat.world.txt', 'pair_landrent.world.txt', 'pair_car.world.txt', 'pair_mammal.world.txt',
 'pair_cit.world.txt', 'pair_oxidants.txt', 'pair_attractiveness.women.txt', 'pair_oxygen.txt',
 'pair_ozone.txt', 'pair_mortality.txt', 'pair_house.world.txt', 'pair_homeless.world.txt', 'pair_glps.txt',
 'pair_prf.world.txt', 'pair_fuel.world.txt', 'pair_fish.fertility.txt', 'pair_fat.world.txt']
'''

ds_num=0
tmp0 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
ttb_inds = all_ttb_inds[ds_num]
tally_preds = tallying(tmp0)
tally_acc = np.sum(tally_preds==tmp0.iloc[:,0])/float(len(tally_preds))
ttb_preds = ttbing(ttb_inds, tmp0)
ttb_acc = np.sum(ttb_preds==tmp0.iloc[:,0])/float(len(ttb_preds))

tally_prior = np.ones((len(ttb_inds),)) #cues have been previously aligned to Y variable
ttb_prior = compute_ttb_prior(ttb_inds)

X_i = tmp0.iloc[:,1:]
Xt_i = np.transpose(X_i)
Y_i = tmp0.iloc[:,0]

lambda_list = np.linspace(0,10000000,num=50)

for lam_bda in lambda_list:
    B_ridge_normal = np.dot(np.linalg.inv(np.dot(Xt_i,X_i) + lam_bda*np.eye(X_i.shape[1])),np.dot(Xt_i,Y_i))
    B_ridge_ttb = np.dot(np.linalg.inv(np.dot(Xt_i,X_i) + lam_bda*np.eye(X_i.shape[1])),(np.dot(Xt_i,Y_i) + lam_bda*ttb_prior))
    B_ridge_tally = np.dot(np.linalg.inv(np.dot(Xt_i,X_i) + lam_bda*np.eye(X_i.shape[1])),(np.dot(Xt_i,Y_i) + lam_bda*tally_prior))
    Y_normalR_preds = np.sign(np.dot(X_i,B_ridge_normal))
    Y_ttbR_preds = np.sign(np.dot(X_i,B_ridge_ttb))
    Y_tallyR_preds = np.sign(np.dot(X_i,B_ridge_tally))
    Y_tallyR_preds = np.sign(np.dot(X_i,tally_prior))
    normalR_acc = np.sum(Y_normalR_preds==Y_i)/float(len(Y_normalR_preds))
    ttbR_acc = np.sum(Y_ttbR_preds==Y_i)/float(len(Y_ttbR_preds))
    tallyR_acc = np.sum(Y_tallyR_preds==Y_i)/float(len(Y_tallyR_preds))

'''
#def ridge_fun(tmp0, lam_bda, prior=False):
for i in [0]: #range(tmp0.shape[0]):
    test_set = tmp0.iloc[i]
    train_set = tmp0.drop(i)
    if prior==False:
        B_ridge = np.dot(np.linalg.inv(np.dot(Xt_i,X_i) + test_lam_bda*np.eye(X_i.shape[1])),np.dot(Xt_i,Y_i))
    else:
        B_ridge = np.dot(np.linalg.inv(np.dot(Xt_i,X_i) + test_lam_bda*np.eye(X_i.shape[1])),(np.dot(Xt_i,Y_i) + test_lam_bda*prior_betas))
'''


'''
train_set, qvalis, ttb_inds, flippers = compute_qvalis(train_set)
tally_up = np.sum(test_set.iloc[:,1:],axis=1)
tally_up = np.array(tally_up/np.abs(tally_up))
np.sum(np.isnan(tally_up))
'''
