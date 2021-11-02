import numpy as np
import os
import pandas as pd
import random
from scipy.optimize import minimize_scalar, minimize
from scipy import stats
import matplotlib.pyplot as plt
import time
import cProfile
import pickle
import time
from scipy.stats import entropy
from sklearn.decomposition import PCA

def q_validities(pair_df):
    cue_valies=[]
    for col in range(1,pair_df.shape[1]):
        pair_df_xitmp = pair_df.iloc[:,col]
        pair_df_ytmp = pair_df.iloc[:,0]
        pair_df_ytmp = pair_df_ytmp[pair_df_xitmp!=0]
        pair_df_xitmp = pair_df_xitmp[pair_df_xitmp!=0]
        cue_vali = np.sum(pair_df_ytmp==pair_df_xitmp)/(float(len(pair_df_ytmp)) + 0.0000000000000001)
        cue_valies.append(cue_vali)
    return np.array(cue_valies)

def compute_qvalis(pair_df, flip_direction=0):
    ds = pair_df.copy()
    #assumes first column of pair_df is Y values
    flippers=[]
    if flip_direction==1:
        chng_q_valis = q_validities(ds)<0.5
        for col in range(1,ds.shape[1]):
            flipper=1 #aligned with Y
            if chng_q_valis[col-1]:
                ds.iloc[:,col] = ds.iloc[:,col]*-1
                flipper=-1 #not aligned with Y
            flippers.append(flipper)
    qvalis = q_validities(ds)
    ttb_inds = np.argsort(qvalis)+1
    ttb_inds = ttb_inds[::-1]
    return ds, ttb_inds, qvalis, flippers

def paired_data(ds_i):
    tmp0 = pd.read_csv(data_dir + ds_fn_list[ds_i], sep="\s+|\t+")
    tmp = tmp0.reindex(np.random.permutation(tmp0.index))
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

def tallying(ds_tmp, tally_dirs):
    ds = ds_tmp.copy()
    if tally_dirs.size==0:
        chng_tdirs = q_validities(ds)<0.5
        tally_dirs = np.ones((ds.shape[1]-1,))
        tally_dirs[chng_tdirs] = -1
    else:
        chng_tdirs = (tally_dirs*-1)+1
    for col in range(1,ds.shape[1]):
        if chng_tdirs[col-1]:
            ds.iloc[:,col] = ds.iloc[:,col]*-1
    tally_up = np.sum(ds.iloc[:,1:],axis=1)
    f = minimize_scalar(lambda B: scaling_prior_log(ds.iloc[:,0], ds.iloc[:,1:], B),
     method='golden', options={'maxiter': 10000})
    scaling = f.x
    tally_dirs = tally_dirs*scaling
    tally_preds = np.sign(tally_up)
    return tally_preds, tally_dirs

def replace_zeros(tally_up):
    tally_up = np.array(tally_up)
    for i in range(len(tally_up)):
        if tally_up[i]==0:
            coin_flip = random.choice([1,-1])
            tally_up[i] = coin_flip
    return tally_up

def scaling_prior(y, x, B):
    B_vec = np.tile(B,x.shape[1])
    y_pred = np.dot(x,B_vec)
    SSE = np.sum(np.power(y-y_pred,2))
    return SSE

def scaling_prior_log(y, x, B):
    B_vec = np.tile(B,x.shape[1])
    preds, _ = g_inv(x,B_vec)
    y_pred = preds
    y = np.array(y)
    y[y==-1] = 0
    SSE = np.sum(np.power(y-y_pred,2))
    return SSE

def get_acc(y,ypreds):
    #expects values to be 1 or -1; 0 for coin flip
    rnd_preds = np.sum(np.array(ypreds) == 0)
    if rnd_preds > 0:
        rnd_acc_premium = rnd_preds*0.5
    else:
        rnd_acc_premium = 0
    try:
        acc = (np.sum(y==ypreds) + rnd_acc_premium)  /float(len(y))
    except:
        acc = 0
    return acc

def ttbing(ds_tmp, ttb_inds, qvalis, flippers):
    if qvalis.size==0:
        ds,ttb_inds,qvalis,flippers = compute_qvalis(ds, flip_direction=1)
    else:
        ds = ds_tmp.copy()
    ttb_preds = np.zeros((ds.shape[0],))
    for i in range(len(ttb_inds)):
        x_i = np.array(ds.iloc[:,ttb_inds[i]]*flippers[ttb_inds[i]-1])
        msk = ttb_preds!=0
        x_i[msk] = 0
        ttb_preds += x_i
    #print(ttb_preds)
    #exit()
    return ttb_preds, ttb_inds

def compute_ttb_prior0(ttb_inds,flippers):
    ttb_prior = np.zeros((len(ttb_inds),))
    reverse_ttb_inds = ttb_inds[::-1] - 1 #order from worst to best cue & 0-based
    for n in range(len(ttb_inds)):
        ttb_prior[reverse_ttb_inds[n]] = 2**n
    ttb_prior = ttb_prior*flippers
    return ttb_prior

def run_ridges(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior0,ttb_prior,tally_prior,ols_betas,permute_ols=0,optim_lambda=0):
    intercept=0
    if intercept==1:
        X_i = np.hstack([np.ones((X_train.shape[0],1)), X_train])
        X_i_ttb = np.hstack([np.ones((X_train.shape[0],1)), X_train_ttb])
        X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])
        X_test_ttb = np.hstack([np.ones((X_test.shape[0],1)), X_test_ttb])
        lambda_mat = lam_bda*np.eye(X_i.shape[1])
        lambda_mat[0,0] = 0
        ttb_prior = np.hstack([1,ttb_prior])
        tally_prior = np.hstack([1,tally_prior])
    else:
        X_i = X_train
        X_i_ttb = X_train_ttb
        lambda_mat = lam_bda*np.eye(X_i.shape[1])
    Xt_i = np.transpose(X_i)
    Xt_i_ttb = np.transpose(X_i_ttb)
    if permute_ols:
        ridge_prior = ols_betas.copy()
        #print(ridge_prior)
        #ridge_prior = np.flip(ridge_prior)
        #print(ridge_prior)
        np.random.shuffle(ridge_prior)
        _, log_normal_acc, _ = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,ridge_prior)
        return log_normal_acc
    else:
        if split_train:
            ridge_prior = ols_betas.copy()
        else:
            ridge_prior = np.zeros((len(ttb_prior),))
    if optim_lambda==0:
        #print('tallying prior')
        B_logR_tally, log_tally_acc, log_tally_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,tally_prior)
        if ols_as_prior:
            print('switched tally for OLS prior: ', B_logR_tally)
        log_tally_entropy = entropy(np.abs(B_logR_tally), base=2) / entropy(np.ones(len(B_logR_tally)), base=2)
        #print('ridge')
        B_logR_normal, log_normal_acc, log_normal_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,ridge_prior)
        log_normal_entropy = entropy(np.abs(B_logR_normal), base=2) / entropy(np.ones(len(B_logR_normal)), base=2)
        #print('ttb prior')
        B_logR_ttb, log_ttb_acc, log_ttb_preds = log_regress_p(X_i_ttb,X_test_ttb,Y_train,Y_test,lam_bda,ttb_prior)
        log_ttb_entropy = entropy(np.abs(B_logR_ttb) * np.abs(ttb_prior0), base=2) / entropy(np.ones(len(B_logR_ttb)), base=2)
        all_preds = [log_tally_preds, log_normal_preds, log_ttb_preds]
        all_entropies = [log_tally_entropy, log_normal_entropy, log_ttb_entropy]
        #print('TAL prior: ', B_logR_tally, tally_prior, log_tally_entropy)
        #print('TTB prior: ', B_logR_ttb, ttb_prior, ttb_prior0, np.abs(B_logR_ttb) * np.abs(ttb_prior0), log_ttb_entropy)
        #print('Zero prior: ', B_logR_normal, log_normal_entropy)
        #print('Zero prior: ', lam_bda, log_normal_preds)
        return log_normal_acc, log_ttb_acc, log_tally_acc, all_preds, all_entropies
    elif optim_lambda=='TAL':
        _, log_tally_acc, _ = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,tally_prior)
        return log_tally_acc
    elif optim_lambda=='TTB':
        _, log_ttb_acc, _ = log_regress_p(X_i_ttb,X_test_ttb,Y_train,Y_test,lam_bda,ttb_prior)
        return log_ttb_acc

def run_ridges2(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior0,ttb_prior,tally_prior,ols_betas,permute_ols=0,optim_lambda=0):
    intercept=0
    if intercept==1:
        X_i = np.hstack([np.ones((X_train.shape[0],1)), X_train])
        X_i_ttb = np.hstack([np.ones((X_train.shape[0],1)), X_train_ttb])
        X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])
        X_test_ttb = np.hstack([np.ones((X_test.shape[0],1)), X_test_ttb])
        lambda_mat = lam_bda*np.eye(X_i.shape[1])
        lambda_mat[0,0] = 0
        ttb_prior = np.hstack([1,ttb_prior])
        tally_prior = np.hstack([1,tally_prior])
    else:
        X_i = X_train
        X_i_ttb = X_train_ttb
        lambda_mat = lam_bda*np.eye(X_i.shape[1])
    Xt_i = np.transpose(X_i)
    Xt_i_ttb = np.transpose(X_i_ttb)
    if permute_ols:
        ridge_prior = ols_betas.copy()
        #print(ridge_prior)
        #ridge_prior = np.flip(ridge_prior)
        #print(ridge_prior)
        np.random.shuffle(ridge_prior)
        _, log_normal_acc, _ = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,ridge_prior)
        return log_normal_acc
    else:
        if split_train:
            ridge_prior = ols_betas.copy()
        else:
            ridge_prior = np.zeros((len(ttb_prior),))
    if optim_lambda==0:
        #print('tallying prior')
        B_logR_tally, log_tally_acc, log_tally_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,tally_prior)
        if ols_as_prior:
            print('switched tally for OLS prior: ', B_logR_tally)
        log_tally_entropy = entropy(np.abs(B_logR_tally), base=2) / entropy(np.ones(len(B_logR_tally)), base=2)
        #print('ridge')
        B_logR_normal, log_normal_acc, log_normal_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,ols_betas.copy())
        B_logR_zero, log_zero_acc, log_zero_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,np.zeros((len(ttb_prior),)))
        log_normal_entropy = entropy(np.abs(B_logR_normal), base=2) / entropy(np.ones(len(B_logR_normal)), base=2)
        #print('ttb prior')
        B_logR_ttb, log_ttb_acc, log_ttb_preds = log_regress_p(X_i_ttb,X_test_ttb,Y_train,Y_test,lam_bda,ttb_prior)
        log_ttb_entropy = entropy(np.abs(B_logR_ttb) * np.abs(ttb_prior0), base=2) / entropy(np.ones(len(B_logR_ttb)), base=2)
        all_preds = [log_tally_preds, log_normal_preds, log_ttb_preds]
        all_entropies = [log_tally_entropy, log_normal_entropy, log_ttb_entropy]
        #print('TAL prior: ', B_logR_tally, tally_prior, log_tally_entropy)
        #print('TTB prior: ', B_logR_ttb, ttb_prior, ttb_prior0, np.abs(B_logR_ttb) * np.abs(ttb_prior0), log_ttb_entropy)
        #print('Zero prior: ', B_logR_normal, log_normal_entropy)
        #print('Zero prior: ', lam_bda, log_normal_preds)
        return log_normal_acc, log_ttb_acc, log_tally_acc, log_zero_acc, all_preds, all_entropies
    elif optim_lambda=='TAL':
        _, log_tally_acc, _ = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,tally_prior)
        return log_tally_acc
    elif optim_lambda=='TTB':
        _, log_ttb_acc, _ = log_regress_p(X_i_ttb,X_test_ttb,Y_train,Y_test,lam_bda,ttb_prior)
        return log_ttb_acc

def sigmoid(x):
    #"Numerically stable sigmoid function."
    #https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)

def pos_sigmoid(x):
    #above or equal to 0
    z = np.exp(-x)
    return 1 - (1 / (1 + z))

def neg_sigmoid(x):
    #below 0
    z = np.exp(x)
    return 1 - (z / (1 + z))

def g_inv(x,b):
    warning = 0
    term = -np.dot(x,b)
    #print('TERM: ', term)
    #for term_i in term:
    #    if term_i > 700:
    #        term_i = 700
    #        print('betas getting big, overflow risk')
    #        warning = 1
    #result_prev = 1/(1+np.exp(term))
    pos_term_bool = term>=0
    pos_term = term.copy()
    neg_term_bool = pos_term_bool==0
    neg_term = term.copy()
    pos_term[neg_term_bool] = np.nan
    pos_result = pos_sigmoid(pos_term)
    #print(pos_result)
    #exit()
    neg_term[pos_term_bool] = np.nan
    neg_result = neg_sigmoid(neg_term)
    #print(neg_result)
    if np.sum(neg_term_bool)>0:
        #print(len(pos_result),len(neg_term_bool),len(neg_result))
        pos_result[neg_term_bool] = neg_result[neg_term_bool]
        result = pos_result
    else:
        neg_result[pos_term_bool] = pos_result[pos_term_bool]
        result = neg_result
    #print(result[0:10])
    #print(result_prev[0:10])
    #time.sleep(3)
    #result = sigmoid(term)
    return result, warning

def beta_thresh(betas):
    beta_test=[]
    for beta in betas:
        beta_test.append(beta>1000000)
    return beta_test

def log_regress_p(X_train,X_test,Y_train,Y_test,lam_bda,beta_zero):
    #print(beta_zero)
    if permute_prior:
        beta_zero = np.random.choice([1,-1],len(beta_zero))*beta_zero
        np.random.shuffle(beta_zero)
    #print(beta_zero)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_train[Y_train==-1] = 0
    X = np.array(X_train)
    Xt = np.transpose(X)
    lambda_mat = lam_bda*np.eye(X.shape[1])
    #beta_old = beta_zero + beta_start #np.random.normal(0,0.1,size=beta_zero.shape) #let's initialize with the prior + noise
    beta_old = beta_start
    beta_new = beta_old*100
    diff_betas = np.sum(np.power(beta_new-beta_old,2))
    #print('ginv1')
    iters = 0
    while diff_betas>0.001 and iters!=2000:
        pseudo_preds, warning = g_inv(X,beta_old)
        if warning==1:
            break
        W = np.diag(np.power(pseudo_preds, 2))
        XtW = np.multiply(Xt,np.diag(W))
        V = np.dot(XtW,X) + lambda_mat
        V_inv = np.linalg.pinv(V)
        pseudo_errs = Y_train - pseudo_preds
        interim_term = np.dot(Xt,pseudo_errs) - (lam_bda*(beta_old-beta_zero))
        beta_new = beta_old + np.dot(V_inv, interim_term )
        diff_betas = np.sum(np.power(beta_new-beta_old,2))
        beta_old = beta_new
        iters += 1
        #beta_test = beta_thresh(beta_new)
        #print(beta_old)
        #time.sleep(0.1)
        if np.mean(np.abs(beta_old))>10:
            #print('big betas, perfect separation probable')
            break
    #print('ginv2')
    pred_probs, _ = g_inv(X_test,beta_new)
    #print('ginv3')
    #print('lambda: ', lam_bda, 'beta_zero: ', beta_zero, 'beta_new: ', beta_new)
    #print('PRED PROBS',pred_probs)
    preds = np.array(pred_probs>0.5,dtype=int)
    preds[preds==0] = -1
    #preds[pred_probs==0.5] = 0
    msk1 = pred_probs<.50000001
    msk2 = pred_probs>.49999999
    msk = [a and b for a, b in zip(msk1, msk2)]
    preds[msk] = 0
    #print(np.sum(pred_probs[msk]==0.5), 'new model guess')
    #print(Y_test[msk].shape, pred_probs[msk].shape, X_test[msk].shape)
    #print(np.sum(preds==0))
    #print(pred_probs)
    #print(msk)
    #exit()
    #preds[msk] = 0
    #print(np.sum(pred_probs==0.5), '0.5')
    acc = get_acc(Y_test,preds)
    return beta_new, acc, preds

def log_regress(X_train,X_test,Y_train,Y_test):
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_train[Y_train==-1] = 0
    X = np.array(X_train)
    Xt = np.transpose(X)
    beta_old = beta_start #np.random.normal(0,0.1,size=X.shape[1]) #let's initialize with noise
    beta_new = beta_old*100
    diff_betas = np.sum(np.power(beta_new-beta_old,2))
    #print('ginv_a')
    iters = 0
    while diff_betas>0.001 and iters!=2000:
        pseudo_preds, warning = g_inv(X,beta_old)
        if warning==1:
            break
        W = np.diag(np.power(pseudo_preds,2))
        XtW = np.multiply(Xt,np.diag(W))
        V = np.dot(XtW,X)
        V_inv = np.linalg.pinv(V)
        pseudo_errs = Y_train - pseudo_preds
        interim_term = np.dot(Xt,pseudo_errs)
        beta_new = beta_old + np.dot(V_inv, interim_term )
        diff_betas = np.sum(np.power(beta_new-beta_old,2))
        beta_old = beta_new
        iters += 1
        #print(beta_old)
        #time.sleep(0.5)
        #beta_test = beta_thresh(beta_new)
        if np.mean(np.abs(beta_old))>10:
            #print('big betas, perfect separation probable')
            break
    #print('ginv_b')
    pred_probs, _ = g_inv(X_test,beta_new)
    #print('ginv_c')
    preds = np.array(pred_probs>0.5,dtype=int)
    preds[preds==0] = -1
    #preds[pred_probs==0.5] = 0
    msk1 = pred_probs<.50000001
    msk2 = pred_probs>.49999999
    msk = [a and b for a, b in zip(msk1, msk2)]
    preds[msk] = 0
    #print(np.sum(pred_probs==0.5), '0.5')
    acc = get_acc(Y_test,preds)
    return acc, preds, beta_new

def regress(X_train,Y_train,X_test,Y_test):
    X_i = X_train
    Xt_i = np.transpose(X_i)
    B_OLS = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i)),np.dot(Xt_i,Y_train))
    Y_OLS_preds = np.sign(np.dot(X_test,B_OLS))
    #Y_OLS_preds = replace_zeros(Y_OLS_preds)
    OLS_acc = get_acc(Y_test,Y_OLS_preds)
    return OLS_acc

def mk_triplets(vals):
    triplets = []
    for i in vals:
        for j in vals:
            for k in vals:
                if i+j+k==0:
                    continue
                triplets.append([i,j,k])
    triplets = np.array(triplets)
    return triplets

def sim_data():
    test_set = mk_triplets([-1,0,1])
    train_triplets = mk_triplets([0,1])
    train_set = np.vstack([random.choice(train_triplets) for _ in range(20)])
    train_set2 = train_set
    train_pairs = []
    for triplet_i in train_set:
        train_set2 = np.delete(train_set2, 0, axis=0)
        for triplet_j in train_set2:
            train_pairs.append(triplet_i-triplet_j)
    train_pairs = np.vstack(train_pairs)
    betas = np.random.exponential(1/2,3)
    noise = 1
    train_e = np.random.normal(0,noise,size=len(train_pairs))
    test_e = np.random.normal(0,noise,size=len(test_set))
    y_train = np.sign(np.dot(train_pairs,betas.T) + train_e)
    y_test = np.sign(np.dot(test_set,betas.T) + test_e)
    return y_test, y_train, test_set, train_pairs

rnd_seed = 333 #666 #101
np.random.seed(rnd_seed)
random.seed(rnd_seed)
testing = 0 # -2 #        2 #-1 #             1,-1,0,2
permute_prior = 0
do_pca = 0 #1
ols_as_prior = 0 #1 # switches TAL for OLS, redundant with permute_prior?
#permute_ols = 1
remove_zeros = 0
v = 0
niters_total = 1000 #   100 #
optim_lambda = 0 #'TAL' #'TTB' #   needs to be 0 to avoid optimization
sample_size = 50 #100
split_train = 0
sigma = 1.3
#lambda_list = np.linspace(0.0001,0.1,num=50) #1000000000000 to converge
#lambda_list = np.hstack([0, np.geomspace(1,1000,num=50)]) #1000000000000 = 1e+12 to converge
lambda_list = np.hstack([0, np.linspace(0.00001,0.1,num=5), np.linspace(0.2,1,num=5), np.linspace(2,10,num=5), np.linspace(20,100,num=5), np.linspace(200,1000,num=5),
    np.geomspace(2000,1000000,num=5), np.geomspace(2000000,1000000000000,num=5), 1000000000000*1000.0,1000000000000*1000000.0, 1000000000000*1000000000.0])
    #, (1000000000000*1000000000.0)**8 ])

wfh = 1
if wfh ==1:
    pwd = '/home/seb/Dropbox/postdoc/LSS_project'
else:
    pwd = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project'

parent_dir = pwd + '/20_classic_datasets/'
#parent_dir = '/home/seb/Dropbox/postdoc/LSS_project/20_classic_datasets/'
#parent_dir = os.getcwd() + '/'
data_dir = parent_dir + 'data/'
tmp_pair_data_dir = parent_dir + 'tmp_pair_data/'
#ds_fn_list = os.listdir(data_dir)
all_ttb_inds = np.load(tmp_pair_data_dir + 'ttb_inds.npy', allow_pickle=True) #in same order as data_dir when loaded through listdir

ml_case = 'breast-cancer-wisconsin.data'
ml_case_labels = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion ','Single Epithelial Cell Size',
'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

ds_fn_list = ['prf.world.txt', 'fish.fertility.txt', 'fuel.world.txt', 'attractiveness.men.txt', 'landrent.world.txt', 'dropout.txt',
    'attractiveness.women.txt', 'cloud.txt', 'car.world.txt', 'mortality.txt', 'bodyfat.world.txt', 'homeless.world.txt', 'oxygen.txt',
    'ozone.txt', 'mammal.world.txt', 'fat.world.txt', 'glps.txt', 'oxidants.txt', 'cit.world.txt', 'house.world.txt']

#professor salries is 0
#obesity is 15
ds_fn_list2 = ['Professors\' Salaries', 'Fish Fertility', 'Fuel Consumption', 'Attractiveness Men', 'Land Rent', 'High School Dropouts',
    'Attractiveness Women', 'Cloud Rainfall', 'Car Accidents', 'Mortality Rates', 'Body Fat', 'Homelessness', 'Oxygen',
    'Ozone in S.F.', 'Mammals\' Sleep', 'Obesity', 'Biodiversity', 'Oxidants in L.A.', 'City Size', 'House Prices']

if testing==1:
    test_fn = 'test_case_red2.csv'
    #test_fn = 'test_case_red.csv'
    #test_fn = 'test_case.csv'
    ds_fn_list = np.tile(test_fn,20)
elif testing==2:
    test_fn = 'not_testing_mode'
    ds_fn_list = ['simulation']
    ds_fn_list2 = ['simulation']
    sv_fn = 'simulation_190' + 'samples_' + str(niters_total) + 'iters'
elif testing == -1:
    test_fn = 'not_testing_mode'
    ds_fn_list = [ml_case]
    ds_fn_list2 = [ml_case]
    sv_fn = 'breast_cancer_' + str(sample_size) + 'samples_' + str(niters_total) + 'iters'
elif testing==-2:
    test_fn = 'not_testing_mode'
    sv_fn = '20ds_' + str(sample_size) + 'samples_' + str(niters_total) + 'iters'
    ds_i = 5
    ds_fn_list = [ds_fn_list[ds_i]] # ['cloud.txt'] #['cit.world.txt'] #['dropout.txt'] #['attractiveness.women.txt'] #
    ds_fn_list2 = [ds_fn_list2[ds_i]] #['Cloud Rainfall'] #['City Size'] #['High School Dropouts'] #['Attractiveness Women'] #
else:
    test_fn = 'not_testing_mode'
    sv_fn = '20ds_' + str(sample_size) + 'samples_' + str(niters_total) + 'iters'

load_previous = 1
if load_previous:
    sv_fn = '20ds_50samples_1000iters_split_train'
    #sv_fn = '20ds_50samples_1000iters_split'

'''
paired_ds_list = ['pair_attractiveness.men.txt', 'pair_dropout.txt', 'pair_cloud.txt',
 'pair_bodyfat.world.txt', 'pair_landrent.world.txt', 'pair_car.world.txt', 'pair_mammal.world.txt',
 'pair_cit.world.txt', 'pair_oxidants.txt', 'pair_attractiveness.women.txt', 'pair_oxygen.txt',
 'pair_ozone.txt', 'pair_mortality.txt', 'pair_house.world.txt', 'pair_homeless.world.txt', 'pair_glps.txt',
 'pair_prf.world.txt', 'pair_fuel.world.txt', 'pair_fish.fertility.txt', 'pair_fat.world.txt']
'''

def medianize(tmp0,cols):
    for col in range(1,len(cols)):
        tmp_x = np.array(tmp0.iloc[:,col], dtype=int)
        tmp_median = np.median(tmp_x)
        tmp_x[tmp_x<tmp_median] = -1
        tmp_x[tmp_x==tmp_median] = 0
        tmp_x[tmp_x>tmp_median] = 1
        #tmp_x = stats.zscore(tmp_x)
        tmp0.iloc[:,col] = tmp_x
    return tmp0

def pcanize(tmp0,strt_col):
    pca = PCA()
    pca.fit(tmp0.iloc[:,strt_col:])
    tmp0.iloc[:,strt_col:] = pca.transform(tmp0.iloc[:,strt_col:])
    return tmp0

beta_start = 0

def main(niter, ds_fn_list,v=0):
    print('iteration: ' + str(niter+1) + ' of ' + str(niters_total))
    all_ds_entropies=[]
    all_ds_accs=[]
    all_ds_agreements=[]
    all_ols_accs=[]
    all_ttb_accs=[]
    all_tally_accs=[]
    all_permute_ols_accs=[]
    for ds_num in range(len(ds_fn_list)):
        #if ds_num==1:
        #    continue
        if v==1:
            print(ds_fn_list[ds_num])
        if ds_fn_list[ds_num] == test_fn:
            tmp0 = pd.read_csv(parent_dir + test_fn, sep=",")
            #tmp0.iloc[:,1] = tmp0.iloc[:,1]*2
            tmp0.iloc[:,0] = np.sum(tmp0.iloc[:,1:3],axis=1) + (tmp0.iloc[:,3]*-1)
            tmp0 = pd.DataFrame(np.tile(tmp0, (20,1)))
            tmp0.iloc[:,0] += np.random.normal(0,sigma,size=tmp0.iloc[:,0].shape)
            tmp0.iloc[:,0] = np.sign(tmp0.iloc[:,0])
        elif ds_fn_list[ds_num] == ml_case:
            tmp0 = pd.read_csv(tmp_pair_data_dir + ds_fn_list[ds_num], header=None, sep=",", usecols=range(1,11))
            cols = [10,1,2,3,4,5,6,7,8,9]
            tmp0 = tmp0[cols]
            tmp_y = tmp0.iloc[:,0]
            tmp_y[tmp_y==2] = 1
            tmp_y[tmp_y==4] = -1
            tmp0.iloc[:,0] = tmp_y
            for col in range(1,len(cols)):
                tmp_x = np.array(tmp0.iloc[:,col])
                bad_rows = np.where(tmp_x=='?')
                if bad_rows[0].size>0:
                    tmp0.drop(list(bad_rows[0]), inplace=True)
            if do_pca:
                tmp0 = pcanize(tmp0,1)
            tmp0 = medianize(tmp0,cols)
            tmp0 = tmp0.astype(float)
        elif ds_fn_list[ds_num] == 'simulation':
            Y_test, Y_train, X_test, X_train = sim_data()
            if do_pca:
                X_test = pcanize(pd.DataFrame(X_test),0)
                X_train = pcanize(pd.DataFrame(X_train),0)
        else:
            tmp0 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
            if do_pca:
                tmp0 = pcanize(tmp0,1)
        if ds_fn_list[ds_num] != 'simulation':
            tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
            if ds_fn_list[ds_num] == ml_case: #balance classes +1/-1
                tmp0_dels = np.sum(tmp0.iloc[:,0]==1) - np.sum(tmp0.iloc[:,0]==-1)
                tmp0_bool = tmp0.iloc[:,0]==1
                tmp0_bool_inds = np.where(tmp0_bool)
                tmp0_bool_inds = tmp0_bool_inds[0][0:tmp0_dels]
                tmp0.drop(tmp0.index[list(tmp0_bool_inds)], inplace=True)
                tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
        #print(tmp0.mean(axis=0))#;exit()
        #mid_ind = int(tmp0.shape[0]*0.5)
        #print(mid_ind)
        #print(tmp0.head())
        mid_ind=sample_size
        all_entropies=[]
        all_accs=[]
        all_agreements=[]
        tally_accs=[]
        ttb_accs=[]
        ols_accs=[]
        for cv in range(1):
            if ds_fn_list[ds_num] != 'simulation':
                if cv==0:
                    train_set = tmp0.iloc[:mid_ind]
                    test_set = tmp0.iloc[mid_ind:]
                elif cv==1:
                    test_set = tmp0.iloc[:mid_ind]
                    train_set = tmp0.iloc[mid_ind:]
                Y_test = test_set.iloc[:,0]
                Y_train = train_set.iloc[:,0]
                X_test = test_set.iloc[:,1:]
                X_train = train_set.iloc[:,1:]
            else:
                train_set = np.hstack([Y_train.reshape((len(Y_train),1)),X_train])
                train_set = pd.DataFrame(train_set)
                test_set = np.hstack([Y_test.reshape((len(Y_test),1)),X_test])
                test_set = pd.DataFrame(test_set)
            global beta_start
            beta_start = np.random.normal(0,0.1,size=X_train.shape[1])
            print(ds_fn_list[ds_num])
            if split_train:
                split_ind = int(np.round(sample_size/2))
                split_ind_start = split_ind
            else:
                split_ind = sample_size
                split_ind_start = 0
            #print(train_set[:split_ind])
            #exit()
            _, tally_prior = tallying(train_set[:split_ind],np.array([])) #if prior not calculated yet, send empty array
            #if ds_fn_list[ds_num] == test_fn:
                #tally_prior = np.array([1,1,-1]) #use only when testing!
            tally_preds, _ = tallying(test_set,np.sign(tally_prior))
            #print(Y_test,tally_preds,X_test)
            #print(np.sum(tally_preds==0), 'tally guess')
            #print(X_test[tally_preds==0].shape)
            if remove_zeros==1:
                #print(np.sum(tally_preds==0))
                remove_zeros_msk = np.array(tally_preds!=0, dtype=bool)
                tally_preds = tally_preds[remove_zeros_msk]
                Y_test = Y_test.iloc[remove_zeros_msk]
                Y_train = Y_train.iloc[remove_zeros_msk]
                X_test = X_test.iloc[remove_zeros_msk,:]
                X_train = X_train.iloc[remove_zeros_msk,:]
                test_set = test_set.iloc[remove_zeros_msk,:]
                train_set = train_set.iloc[remove_zeros_msk,:]
            tally_acc = get_acc(Y_test,tally_preds)
            #exit()
            tally_accs.append(tally_acc)
            _,ttb_inds,qvalis,flippers = compute_qvalis(train_set[:split_ind], flip_direction=1)
            ttb_preds, _ = ttbing(test_set,ttb_inds,qvalis,flippers)
            #print(np.sum(ttb_preds==0)); exit()
            ttb_acc = get_acc(Y_test,ttb_preds)
            ttb_accs.append(ttb_acc)
            ttb_prior0 = compute_ttb_prior0(ttb_inds,flippers)
            #print('ttb prior 0 (multiples X)')
            #print(ttb_prior0)
            #if permute_prior:
            #    np.random.shuffle(ttb_prior0)
            #print(ttb_prior0)
            ttb_prior0 = ttb_prior0.reshape((ttb_prior0.shape + (1,)))
            #print(X_train[:split_ind].shape[0])
            #exit()
            #ttb_prior0_train = np.tile(ttb_prior0,X_train[:split_ind].shape[0]).T
            ttb_prior0_train = np.tile(ttb_prior0,X_train.shape[0]).T
            ttb_prior0_test = np.tile(ttb_prior0,X_test.shape[0]).T
            #ttb_prior0_normed = ttb_prior0/np.max(np.abs(ttb_prior0_train))
            #X_train_ttb = (ttb_prior0_train[:split_ind]*X_train[:split_ind])/np.max(np.abs(ttb_prior0_train[:split_ind]))
            #X_test_ttb = (ttb_prior0_test*X_test)/np.max(np.abs(ttb_prior0_train[:split_ind]))
            X_train_ttb = (ttb_prior0_train*X_train)/np.max(np.abs(ttb_prior0_train))
            X_test_ttb = (ttb_prior0_test*X_test)/np.max(np.abs(ttb_prior0_train))
            f = minimize_scalar(lambda B: scaling_prior_log(Y_train[:split_ind], X_train_ttb[:split_ind], B),
            method='golden', options={'maxiter': 10000})
            ttb_prior = np.tile(f.x,X_train[:split_ind].shape[1]) #after ttbing X, we need a scaling prior
            ttb_prior_list = [ttb_prior0.flatten(), ttb_prior]
            log_ols_acc, log_ols_preds, ols_betas = log_regress(X_train[:split_ind],X_test,Y_train[:split_ind],Y_test)
            ols_accs.append(log_ols_acc)
            cv_accs=[]
            cv_entropies=[]
            cv_agreement = []
            cv_permute_ols_accs = []
            for lam_bda in lambda_list:
                print('lambda: ',lam_bda)
                #print(tally_prior, ols_betas)
                #exit()
                #print('OLS weights: ',ols_betas)
                #permute_ols = 0
                if ols_as_prior:
                    tally_prior = ols_betas
                if split_train:
                    normalR_acc, ttbR_acc, tallyR_acc, zeroR_acc, preds3, entropies3 = run_ridges2(X_train[split_ind_start:],X_train_ttb[split_ind_start:],Y_train[split_ind_start:],
                    X_test,X_test_ttb,Y_test,lam_bda,ttb_prior0.flatten(),
                    ttb_prior,tally_prior,ols_betas)
                else:
                    normalR_acc, ttbR_acc, tallyR_acc, preds3, entropies3 = run_ridges(X_train[split_ind_start:],X_train_ttb[split_ind_start:],Y_train[split_ind_start:],
                    X_test,X_test_ttb,Y_test,lam_bda,ttb_prior0.flatten(),
                    ttb_prior,tally_prior,ols_betas)
                tallyR_preds, normalR_preds, ttbR_preds = preds3
                tallyR_entropy, normalR_entropy, ttbR_entropy = entropies3
                cv_entropies.append([tallyR_entropy, normalR_entropy, ttbR_entropy])
                if split_train:
                    cv_accs.append([normalR_acc, ttbR_acc, tallyR_acc, zeroR_acc])
                else:
                    cv_accs.append([normalR_acc, ttbR_acc, tallyR_acc])
                agree1 = np.array([get_acc(log_ols_preds,tallyR_preds), get_acc(tally_preds,tallyR_preds), get_acc(log_ols_preds,ttbR_preds), get_acc(ttb_preds,ttbR_preds)])
                agree2 = np.array([get_acc(tallyR_preds,log_ols_preds), get_acc(tallyR_preds,tally_preds), get_acc(ttbR_preds,log_ols_preds), get_acc(ttbR_preds,ttb_preds)])
                cv_agreement.append((agree1+agree2)/2)
                permute_normalR_acc = run_ridges(X_train[split_ind_start:],X_train_ttb[split_ind_start:],Y_train[split_ind_start:],
                X_test,X_test_ttb,Y_test,lam_bda,ttb_prior0.flatten(),
                ttb_prior,tally_prior,ols_betas,permute_ols=1)
                cv_permute_ols_accs.append(permute_normalR_acc)
                #if lam_bda==lambda_list[-1]:
                    #time.sleep(1)
            #exit()
            all_entropies.append(cv_entropies)
            all_accs.append(cv_accs)
            all_agreements.append(cv_agreement)
            #all_permute_ols_accs.append(cv_permute_ols_accs)
            #print 'tally_prior: ', tally_prior
            #exit()
            if v==1:
                print('tally: ', np.mean(tally_accs))
                print('tally_prior: ', tally_prior)
                print('ttb: ', np.mean(ttb_accs))
                print('ttb_prior: ', ttb_prior)
                print('ols: ', np.mean(ols_accs))
                all_accs_means = np.mean(all_accs, axis=0)
                print('max ridge (normal, ttb_prior, tally_prior):', np.max(all_accs_means,axis=0))
                print('--------------------------------------------------------------------')
        #exit()
        all_ols_accs.append(np.array(ols_accs))
        all_ttb_accs.append(np.array(ttb_accs))
        all_tally_accs.append(np.array(tally_accs))
        all_ds_entropies.append(np.array(all_entropies))
        all_ds_accs.append(np.array(all_accs))
        all_ds_agreements.append(np.array(all_agreements))
        all_permute_ols_accs.append(cv_permute_ols_accs)
        #exit()
    return all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs, all_ds_agreements, all_permute_ols_accs, all_ds_entropies

def main_optim(mid_ind, lam_bda, optim_lambda, ds_num, niter):
    all_ds_accs=[]
    all_ds_agreements=[]
    all_ols_accs=[]
    all_ttb_accs=[]
    all_tally_accs=[]
    if ds_fn_list[ds_num] == test_fn:
        tmp0 = pd.read_csv(parent_dir + test_fn, sep=",")
        tmp0.iloc[:,0] = np.sum(tmp0.iloc[:,1:3],axis=1) + (tmp0.iloc[:,3]*-1)
        tmp0 = pd.DataFrame(np.tile(tmp0, (20,1)))
        tmp0.iloc[:,0] += np.random.normal(0,sigma,size=tmp0.iloc[:,0].shape)
        tmp0.iloc[:,0] = np.sign(tmp0.iloc[:,0])
    elif ds_fn_list[ds_num] == ml_case:
        tmp0 = pd.read_csv(tmp_pair_data_dir + ds_fn_list[ds_num], header=None, sep=",", usecols=range(1,11))
        cols = [10,1,2,3,4,5,6,7,8,9]
        tmp0 = tmp0[cols]
        tmp_y = tmp0.iloc[:,0]
        tmp_y[tmp_y==2] = 1
        tmp_y[tmp_y==4] = -1
        tmp0.iloc[:,0] = tmp_y
        for col in range(1,len(cols)):
            tmp_x = np.array(tmp0.iloc[:,col])
            bad_rows = np.where(tmp_x=='?')
            if bad_rows[0].size>0:
                tmp0.drop(list(bad_rows[0]), inplace=True)
        #for col in range(1,len(cols)):
        #    tmp_x = np.array(tmp0.iloc[:,col], dtype=int)
        #    tmp_median = np.median(tmp_x)
        #    tmp_x[tmp_x<tmp_median] = -1
        #    tmp_x[tmp_x==tmp_median] = 0
        #    tmp_x[tmp_x>tmp_median] = 1
        #    #tmp_x = stats.zscore(tmp_x)
        #    tmp0.iloc[:,col] = tmp_x
        tmp0 = medianize(tmp0,cols)
        tmp0 = tmp0.astype(float)
    elif ds_fn_list[ds_num] == 'simulation':
        Y_test, Y_train, X_test, X_train = sim_data()
    else:
        tmp0 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
    if ds_fn_list[ds_num] != 'simulation':
        tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
        if ds_fn_list[ds_num] == ml_case:
            tmp0_dels = np.sum(tmp0.iloc[:,0]==1) - np.sum(tmp0.iloc[:,0]==-1)
            tmp0_bool = tmp0.iloc[:,0]==1
            tmp0_bool_inds = np.where(tmp0_bool)
            tmp0_bool_inds = tmp0_bool_inds[0][0:tmp0_dels]
            tmp0.drop(tmp0.index[list(tmp0_bool_inds)], inplace=True)
            tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
        all_accs=[]
        all_agreements=[]
        tally_accs=[]
        ttb_accs=[]
        ols_accs=[]
        for cv in range(1):
            if ds_fn_list[ds_num] != 'simulation':
                if cv==0:
                    train_set = tmp0.iloc[:mid_ind]
                    test_set = tmp0.iloc[mid_ind:]
                elif cv==1:
                    test_set = tmp0.iloc[:mid_ind]
                    train_set = tmp0.iloc[mid_ind:]
                Y_test = test_set.iloc[:,0]
                Y_train = train_set.iloc[:,0]
                X_test = test_set.iloc[:,1:]
                X_train = train_set.iloc[:,1:]
            else:
                train_set = np.hstack([Y_train.reshape((len(Y_train),1)),X_train])
                train_set = pd.DataFrame(train_set)
                test_set = np.hstack([Y_test.reshape((len(Y_test),1)),X_test])
                test_set = pd.DataFrame(test_set)
            global beta_start
            beta_start = np.random.normal(0,0.1,size=X_train.shape[1])
            log_ols_acc, _, _ = log_regress(X_train,X_test,Y_train,Y_test)
            if optim_lambda=='TAL':
                _, tally_prior = tallying(train_set,np.array([])) #if prior not calculated yet, send empty array
                tally_preds, _ = tallying(test_set,np.sign(tally_prior))
                if remove_zeros==1:
                    remove_zeros_msk = np.array(tally_preds!=0, dtype=bool)
                    tally_preds = tally_preds[remove_zeros_msk]
                    Y_test = Y_test.iloc[remove_zeros_msk]
                    Y_train = Y_train.iloc[remove_zeros_msk]
                    X_test = X_test.iloc[remove_zeros_msk,:]
                    X_train = X_train.iloc[remove_zeros_msk,:]
                    test_set = test_set.iloc[remove_zeros_msk,:]
                    train_set = train_set.iloc[remove_zeros_msk,:]
                tally_acc = get_acc(Y_test,tally_preds)
                ref_acc = tally_acc
                acc_optim = run_ridges(X_train,X_train,Y_train,X_test,X_test,Y_test,lam_bda,tally_prior,tally_prior,optim_lambda=optim_lambda)
            elif optim_lambda=='TTB':
                _,ttb_inds,qvalis,flippers = compute_qvalis(train_set, flip_direction=1)
                ttb_preds, _ = ttbing(test_set,ttb_inds,qvalis,flippers)
                ttb_acc = get_acc(Y_test,ttb_preds)
                ref_acc = ttb_acc
                ttb_prior0 = compute_ttb_prior0(ttb_inds,flippers)
                ttb_prior0 = ttb_prior0.reshape((ttb_prior0.shape + (1,)))
                ttb_prior0_train = np.tile(ttb_prior0,X_train.shape[0]).T
                ttb_prior0_test = np.tile(ttb_prior0,X_test.shape[0]).T
                X_train_ttb = (ttb_prior0_train*X_train)/np.max(np.abs(ttb_prior0_train))
                X_test_ttb = (ttb_prior0_test*X_test)/np.max(np.abs(ttb_prior0_train))
                f = minimize_scalar(lambda B: scaling_prior_log(Y_train, X_train_ttb, B),
                method='golden', options={'maxiter': 10000})
                ttb_prior = np.tile(f.x,X_train.shape[1]) #after ttbing X, we need a scaling prior
                acc_optim = run_ridges(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior,ttb_prior,optim_lambda=optim_lambda)
            #acc_diff = acc_optim - ref_acc
    return acc_optim, ref_acc, log_ols_acc

def main_optim_wrapper(params, optim_lambda, niters_total,ds_num):
    #np.random.seed(rnd_seed+1)
    #random.seed(rnd_seed+1)
    mid_ind, lam_bda = params
    #print(params)
    mid_ind = int(np.round(mid_ind))
    total_acc_optim=[]
    total_ref_acc=[]
    total_ols_accs=[]
    for niter in range(niters_total):
        acc_optim, ref_acc, log_ols_acc = main_optim(mid_ind, lam_bda, optim_lambda, ds_num, niter)
        total_acc_optim.append(acc_optim)
        total_ref_acc.append(ref_acc)
        total_ols_accs.append(log_ols_acc)
    ols_mn = np.mean(total_ols_accs)
    heur_mn = np.mean(total_ref_acc)
    optim_mn = np.mean(total_acc_optim)
    print('ols: ',ols_mn, optim_lambda, heur_mn, 'optim: ', optim_mn, params)
    #np.random.seed(int(np.round(time.time())))
    #random.seed(int(np.round(time.time())))
    if ols_mn > heur_mn and ols_mn > optim_mn:
        mn_acc_diff = mid_ind/10 + np.random.randn(1)/10
    elif ols_mn > heur_mn and ols_mn < optim_mn:
        mn_acc_diff = ols_mn - optim_mn
    else:
        mn_acc_diff = heur_mn - optim_mn #+ np.random.randn(1)/10000
    print('mn_acc_diff: ', mn_acc_diff)
    return mn_acc_diff

if optim_lambda!=0:
    all_f=[]
    if ds_fn_list[0] == ml_case:
        ds_num = 0
        print(ds_fn_list[ds_num], ds_num)
        tmp00 = pd.read_csv(tmp_pair_data_dir + ds_fn_list[ds_num], header=None, sep=",", usecols=range(1,11))
        bnd = tmp00.shape[0]-10
        if bnd > 1000:
            bnd = 1000
        bnds = ((9, bnd), (0, None))
        f = minimize(main_optim_wrapper, (12, 1), args=(optim_lambda, niters_total,ds_num),
        method='SLSQP', options={'ftol': 1e-03, 'disp': True, 'maxiter': 100},bounds=bnds)
        all_f.append(f)
        print(f)
        #pickle.dump( all_f, open( parent_dir + 'f_' + optim_lambda, "wb" ) )
        exit()
    else:
        for ds_num in range(len(ds_fn_list)):
            print(ds_fn_list[ds_num], ds_num)
            tmp00 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
            bnd = tmp00.shape[0]-10
            if bnd > 1000:
                bnd = 1000
            bnds = ((9, bnd), (0, None))
            f = minimize(main_optim_wrapper, (12, 1), args=(optim_lambda, niters_total,ds_num),
            method='SLSQP', options={'ftol': 1e-03, 'disp': True, 'maxiter': 100},bounds=bnds)
            all_f.append(f)
            print(f)
        pickle.dump( all_f, open( parent_dir + 'f_' + optim_lambda, "wb" ) )
        exit()

#TAL<- 0: (16,1), 1: (x,x), 2: (18,1533), 3: (17, 103768), 4: (9, 0.2), 5: (12, 1), 6: (12, 1), 7: (12, 1), 8: (12, 1), 9: ()
#ml_case -> TTB: [12,50] & [80,0.000001], TAL: [12,77]

def parallelizer(niters_total,ds_fn_list,v):
    total_ols_accs=[]
    total_ttb_accs=[]
    total_tally_accs=[]
    total_ds_accs=[]
    total_ds_entropies=[]
    total_ds_agreements=[]
    total_permute_ols_accs=[]
    for niter in range(niters_total):
        all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs, all_ds_agreements, all_permute_ols_accs, all_ds_entropies = main(niter, ds_fn_list, v=v)
        total_ols_accs.append(all_ols_accs)
        total_ttb_accs.append(all_ttb_accs)
        total_tally_accs.append(all_tally_accs)
        total_ds_accs.append(all_ds_accs)
        total_ds_entropies.append(all_ds_entropies)
        total_ds_agreements.append(all_ds_agreements)
        total_permute_ols_accs.append(all_permute_ols_accs)
    return total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements, total_permute_ols_accs, total_ds_entropies

#cProfile.run('parallelizer(niters_total,ds_fn_list,v)')
if load_previous:
    total_data_load = pickle.load( open( parent_dir + sv_fn, "rb" ) )
    total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements, total_permute_ols_accs, total_ds_entropies = total_data_load
else:
    total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements, total_permute_ols_accs, total_ds_entropies = parallelizer(niters_total,ds_fn_list,v)

all_ols_accs = np.mean(total_ols_accs,axis=0)
all_ttb_accs = np.mean(total_ttb_accs,axis=0)
all_tally_accs = np.mean(total_tally_accs,axis=0)
all_ds_accs = np.mean(total_ds_accs,axis=0)

lambda_list2=['OLS','TTB','TAL']
for l in lambda_list:
    lambda_list2.append('{:.1e}'.format(l))

def get_means(ds_num):
    big3 = [np.mean(all_ols_accs[ds_num]), np.mean(all_ttb_accs[ds_num]), np.mean(all_tally_accs[ds_num])]
    lambda_accs = np.mean(all_ds_accs[ds_num], axis=0)
    ridge_normal = lambda_accs[:,0]
    ridge_ttb = lambda_accs[:,1]
    ridge_tally = lambda_accs[:,2]
    if split_train:
        ridge_zero = lambda_accs[:,3]
        return ridge_normal, ridge_ttb, ridge_tally, ridge_zero, big3
    else:
        return ridge_normal, ridge_ttb, ridge_tally, big3

def get_stds(ds_num):
    all_ols_stds = np.std(total_ols_accs,axis=0)#/ np.power(niters_total,1/2)
    all_ttb_stds = np.std(total_ttb_accs,axis=0)#/ np.power(niters_total,1/2)
    all_tally_stds = np.std(total_tally_accs,axis=0)#/ np.power(niters_total,1/2)
    all_ds_stds = np.std(total_ds_accs,axis=0)#/ np.power(niters_total,1/2)
    big3 = [np.mean(all_ols_stds[ds_num]), np.mean(all_ttb_stds[ds_num]), np.mean(all_tally_stds[ds_num])]
    lambda_accs = np.mean(all_ds_stds[ds_num], axis=0)
    ridge_normal = lambda_accs[:,0]
    ridge_ttb = lambda_accs[:,1]
    ridge_tally = lambda_accs[:,2]
    if split_train:
        ridge_zero = lambda_accs[:,3]
        return ridge_normal, ridge_ttb, ridge_tally, ridge_zero, big3
    else:
        return ridge_normal, ridge_ttb, ridge_tally, big3

'''
def get_plot():
    plt.rcParams["figure.figsize"] = (20,10)
    if testing == -1:
        fig, ax = plt.subplots(nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(nrows=4, ncols=5)
    i = 0
    for row in ax:
        for col in row:
            ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(i)
            #y = np.hstack([big3, ridge_betas])
            col.plot(0, big3[0], 'bo', label='OLS') #ols
            col.plot(range(len(big3)-2, len(big3)-2 + len(ridge_normal)), ridge_normal, '-b', label="Zero prior")
            col.plot(range(len(big3)-2, len(big3)-2 + len(ridge_normal)), ridge_ttb, '-g', label='TTB prior')
            col.plot(range(len(big3)-2, len(big3)-2 + len(ridge_normal)), ridge_tally, '-r', label='TAL prior')
            col.plot(len(ridge_normal)+1, big3[1], 'g*', label='TTB') #ttb
            col.plot(len(ridge_normal)+1, big3[2], 'r*', label='TAL') #tally
            col.set_title(ds_fn_list2[i])
            col.set_xlabel('Model/Penalty parameter')
            col.set_ylabel('Accuracy')
            if testing != -1:
                i += 1
    plt.subplots_adjust( hspace=0.36, wspace=0.2 , top=0.94 , left=0.07 , right=0.96 , bottom=0.03 )
    plt.show()
'''


def get_plot():
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 10})
    plt.rc('xtick',labelsize=6)
    plt.rc('ytick',labelsize=6)
    fig, ax = plt.subplots(nrows=4, ncols=5)
    fig.canvas.draw()
    i = 0
    for row in ax:
        for col in row:
            if i > 19:
                continue
            if split_train:
                ridge_normal, ridge_ttb, ridge_tally, ridge_zero, big3 = get_means(i)
                ridge_normal_std, ridge_ttb_std, ridge_tally_std, ridge_zero_std, big3_std = get_stds(i)
            else:
                ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(i)
                ridge_normal_std, ridge_ttb_std, ridge_tally_std, big3_std = get_stds(i)
            if split_train:
                col.plot(0, big3[0], 'yo', label='OLS') #ols
            else:
                col.plot(0, big3[0], 'bo', label='OLS') #ols
            #col.plot(1, big3[1], 'go', label='TTB') #ttb
            #col.plot(2, big3[2], 'ro', label='TAL') #tally
            col.plot(len(ridge_normal), big3[1], 'g*', label='TTB') #ttb
            col.plot(len(ridge_normal), big3[2], 'r*', label='TAL') #tally
            if split_train:
                col.plot(lambda_list2[3:], ridge_zero, '-b', label="Zero prior")
                col.plot(lambda_list2[3:], ridge_normal, '-y', label="OLS prior")
            else:
                col.plot(lambda_list2[3:], ridge_normal, '-b', label="Zero prior")
            col.plot(lambda_list2[3:], ridge_ttb, '-g', label="TTB prior")
            if ols_as_prior:
                col.plot(lambda_list2[3:], ridge_tally, '-k', label="OLS prior")
            else:
                col.plot(lambda_list2[3:], ridge_tally, '-r', label="TAL prior")
            #col.plot(lambda_list2[3+np.argmax(ridge_tally)], ridge_tally[np.argmax(ridge_tally)], '*r', label="TAL prior *")
            #col.plot(lambda_list2[3+np.argmax(ridge_ttb)], ridge_ttb[np.argmax(ridge_ttb)], '*g', label="TTB prior *")
            #col.plot(lambda_list2[3+np.argmax(ridge_normal)], ridge_normal[np.argmax(ridge_normal)], '*b', label="Zero prior *")
            col.set_title(ds_fn_list2[i])
            shrink_std = 0.5
            if split_train:
                col.fill_between(lambda_list2[3:], ridge_normal - (ridge_normal_std*shrink_std), #ridge_conf[0], #
                ridge_normal + (ridge_normal_std*shrink_std), alpha=0.1, color='y') #ridge_conf[1]
                col.fill_between(lambda_list2[3:], ridge_zero - (ridge_zero_std*shrink_std), #ridge_conf[0], #
                ridge_zero + (ridge_zero_std*shrink_std), alpha=0.1, color='b') #ridge_conf[1]
            else:
                col.fill_between(lambda_list2[3:], ridge_normal - (ridge_normal_std*shrink_std), #ridge_conf[0], #
                ridge_normal + (ridge_normal_std*shrink_std), alpha=0.1, color='b') #ridge_conf[1]
            col.fill_between(lambda_list2[3:], ridge_ttb - (ridge_ttb_std*shrink_std),
            ridge_ttb + (ridge_ttb_std*shrink_std), alpha=0.1, color='g')
            col.fill_between(lambda_list2[3:], ridge_tally - (ridge_tally_std*shrink_std),
            ridge_tally + (ridge_tally_std*shrink_std), alpha=0.1, color='r')
            j = 0
            for tick in col.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
                if i<15:
                    tick.set_visible(False)
                #else:
                #    if j%2:
                #        tick.set_visible(False)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                tick.label.set_rotation(60)
                j += 1
            if i==4:
                col.legend(loc='lower left', prop={'size': 4.3})
            #print(np.argmax(ridge_tally), np.argmax(ridge_ttb))
            #col.set_xlabel('Model/Penalty parameter', fontsize=12)
            #col.set_ylabel('Accuracy', fontsize=16)
            i += 1
    #print(np.shape(ridge_tally), np.shape(ridge_ttb))
    #exit()
    plt.subplots_adjust( hspace=0.5, wspace=0.344 , top=0.94 , left=0.079 , right=0.89 , bottom=0.113 )
    i = 0
    for ax in fig.axes:
        N_y = 5
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.round(np.linspace(ymin, ymax, N_y), 2))
        N_x = 8
        if i==19:
            N_x = 6
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.round(np.linspace(xmin, xmax, N_x), 2))
        #plt.sca(ax)
        #plt.xticks(rotation=60)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(45)
        i += 1
    # Set common labels
    fig.text(0.5, 0.02, 'Penalty parameter', ha='center', va='center', fontsize=18)
    fig.text(0.03, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=18)
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #fig.savefig(parent_dir + '20_ds_pngs/' + fn_20classic + '.png', bbox_inches='tight')
    plt.show()

#get_plot()
#plt.close()

def get_entropies():
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 10})
    if testing == -1 or testing == -2:
        fig, ax = plt.subplots(nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(nrows=4, ncols=5)
    i = 0
    for row in ax:
        for col in row:
            #print(i)
            tally_entropies=[]
            ols_entropies=[]
            ttb_entropies=[]
            for niter in range(niters_total):
                tally_entropies.append(total_ds_entropies[niter][i][0][:,0])
                ols_entropies.append(total_ds_entropies[niter][i][0][:,1])
                ttb_entropies.append(total_ds_entropies[niter][i][0][:,2])
            ols_entropies2 = np.mean(ols_entropies, axis=0)
            tally_entropies2 = np.mean(tally_entropies, axis=0)
            ttb_entropies2 = np.mean(ttb_entropies, axis=0)
            shrink_std = 2
            ols_entropies2_stds = np.std(ols_entropies, axis=0) * shrink_std
            tally_entropies2_stds = np.std(tally_entropies, axis=0) * shrink_std
            ttb_entropies2_stds = np.std(ttb_entropies, axis=0) * shrink_std
            col.plot(0, ols_entropies2[0], 'bo', label='OLS') #ols
            col.plot(lambda_list2[3:], ols_entropies2, '-b', label="Zero prior")
            col.plot(lambda_list2[3:], ttb_entropies2, '-g', label='TTB prior')
            col.plot(lambda_list2[3:], tally_entropies2, '-r', label='TAL prior')
            col.plot(len(lambda_list2[3:]), ttb_entropies2[-1], 'g*', label='TTB') #ttb
            col.plot(len(lambda_list2[3:]), tally_entropies2[-1], 'r*', label='TAL') #tally
            col.set_title(ds_fn_list2[i])
            col.set_ylim(ymax = 1.05, ymin = 0)
            col.fill_between(lambda_list2[3:], ols_entropies2 - ols_entropies2_stds, #ridge_conf[0], #
            ols_entropies2 + ols_entropies2_stds, alpha=0.1, color='b') #ridge_conf[1]
            col.fill_between(lambda_list2[3:], ttb_entropies2 - ttb_entropies2_stds,
            ttb_entropies2 + ttb_entropies2_stds, alpha=0.1, color='g')
            col.fill_between(lambda_list2[3:], tally_entropies2 - tally_entropies2_stds,
            tally_entropies2 + tally_entropies2_stds, alpha=0.1, color='r')
            #col.set_xlabel('Model/Penalty parameter')
            #col.set_ylabel('Normalized \n entropy')
            #col.set_xticklabels(lambda_list2[3:], rotation=40)
            j = 0
            for tick in col.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
                if i<15:
                    tick.set_visible(False)
                else:
                    if j%2:
                        tick.set_visible(False)
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small')
                tick.label.set_rotation(60)
                j += 1
            if i==4:
                col.legend(loc='lower right', prop={'size': 4.5})
            if testing != -1 or testing != -2:
                i += 1
    fig.text(0.5, 0.02, 'Penalty parameter', ha='center', va='center', fontsize=18)
    fig.text(0.03, 0.5, 'Normalized entropy', ha='center', va='center', rotation='vertical', fontsize=18)
    plt.subplots_adjust( hspace=0.36, wspace=0.3 , top=0.94 , left=0.08 , right=0.96 , bottom=0.145 )
    plt.show()

#get_entropies()

def get_plot2(title, i, shrink_std):
    #from scipy import stats
    plt.rcParams.update({'font.size': 10})
    ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(i)
    ridge_normal_std, ridge_ttb_std, ridge_tally_std, big3_std = get_stds(i)
    plt.rcParams["figure.figsize"] = (10,10)
    fig, ax = plt.subplots()
    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()
    ymax = np.max([ridge_normal.max(), ridge_ttb.max(), ridge_tally.max(), np.max(big3)])
    ax.plot(0, big3[0], 'bo', label='OLS') #ols
    ax.plot(lambda_list2[3:], ridge_normal, '-b', label="Zero prior")
    ax.plot(lambda_list2[3:], ridge_ttb, '-g', label="TTB prior")
    ax.plot(lambda_list2[3:], ridge_tally, '-r', label="TAL prior")
    ax.plot(lambda_list2[-1], big3[1], '*g', label='TTB') #ttb
    ax.plot(lambda_list2[-1], big3[2], 'r*', label='TAL') #tally
    #ridge_conf = stats.norm.interval(0.68, loc=ridge_normal, scale=ridge_normal_std/np.sqrt(niters_total))
    #print(ridge_conf)
    #shrink_std = 1
    ax.fill_between(lambda_list2[3:], ridge_normal - (ridge_normal_std*shrink_std), #ridge_conf[0], #
    ridge_normal + (ridge_normal_std*shrink_std), alpha=0.1, color='b') #ridge_conf[1]
    ax.fill_between(lambda_list2[3:], ridge_ttb - (ridge_ttb_std*shrink_std),
    ridge_ttb + (ridge_ttb_std*shrink_std), alpha=0.1, color='g')
    ax.fill_between(lambda_list2[3:], ridge_tally - (ridge_tally_std*shrink_std),
    ridge_tally + (ridge_tally_std*shrink_std), alpha=0.1, color='r')
    #ax.set_title("Breast Cancer Dataset")
    #ax.set_ylim(ymax = 1, ymin = 0)
    ax.set_title(title)
    #ax.set_xticklabels(lambda_list2, rotation=45)
    plt.xticks(rotation=60)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.xlabel('Penalty parameter', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    #plt.figure(num=1, figsize=(20,20), dpi=300, facecolor='w', edgecolor='k')
    #fig.savefig(parent_dir + 'breast_cancer_data.png', bbox_inches='tight', dpi=300)
    plt.show()

i = 0
#get_plot2(ds_fn_list2[i] + ' (N = ' + str(sample_size) + ')', i)
#get_plot2(ds_fn_list2[i], i)

def get_entropy2(title, i, shrink_std):
    ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(i)
    plt.rcParams["figure.figsize"] = (10,10)
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots()
    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()
    tally_entropies=[]
    ols_entropies=[]
    ttb_entropies=[]
    for niter in range(niters_total):
        tally_entropies.append(total_ds_entropies[niter][i][0][:,0])
        ols_entropies.append(total_ds_entropies[niter][i][0][:,1])
        ttb_entropies.append(total_ds_entropies[niter][i][0][:,2])
    ols_entropies2 = np.mean(ols_entropies, axis=0)
    tally_entropies2 = np.mean(tally_entropies, axis=0)
    ttb_entropies2 = np.mean(ttb_entropies, axis=0)
    #shrink_std = 2
    ols_entropies2_stds = np.std(ols_entropies, axis=0) * shrink_std
    tally_entropies2_stds = np.std(tally_entropies, axis=0) * shrink_std
    ttb_entropies2_stds = np.std(ttb_entropies, axis=0) * shrink_std
    ax.plot(0, ols_entropies2[0], 'bo', label='OLS') #ols
    ax.plot(lambda_list2[3:], ols_entropies2, '-b', label="Zero prior")
    ax.plot(lambda_list2[3:], ttb_entropies2, '-g', label='TTB prior')
    ax.plot(lambda_list2[3:], tally_entropies2, '-r', label='TAL prior')
    ax.plot(len(lambda_list2[3:])-1, ttb_entropies2[-1], 'g*', label='TTB') #ttb
    ax.plot(len(lambda_list2[3:])-1, tally_entropies2[-1], 'r*', label='TAL') #tally
    ax.fill_between(lambda_list2[3:], ols_entropies2 - ols_entropies2_stds, #ridge_conf[0], #
    ols_entropies2 + ols_entropies2_stds, alpha=0.1, color='b') #ridge_conf[1]
    ax.fill_between(lambda_list2[3:], ttb_entropies2 - ttb_entropies2_stds,
    ttb_entropies2 + ttb_entropies2_stds, alpha=0.1, color='g')
    ax.fill_between(lambda_list2[3:], tally_entropies2 - tally_entropies2_stds,
    tally_entropies2 + tally_entropies2_stds, alpha=0.1, color='r')
    #ax.set_title(ds_fn_list2[i])
    ax.set_ylim(ymax = 1.05, ymin = 0)
    ax.set_title(title)
    #ax.set_xticklabels(lambda_list2, rotation=45)
    plt.xticks(rotation=60)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.xlabel('Penalty parameter', fontsize=18)
    plt.ylabel('Normalized entropy', fontsize=16)
    plt.show()

#i = 0 #13
#get_entropy2(ds_fn_list2[i] + ' (N = ' + str(sample_size) + ')', i)
#get_entropy2(ds_fn_list2[i], i)

def get_worst_best(data, ds_num, accs=1):
    if accs:
        total_ds_accs, total_permute_ols_accs = data
        cols = [0,1,2]
        if len(total_permute_ols_accs[0])>1:
            ridge_normal_permute_tmp=[]
            for i in range(niters_total):
                ridge_normal_permute_tmp.append(total_permute_ols_accs[i][ds_num])
            ridge_normal_permute = np.mean(np.vstack(ridge_normal_permute_tmp), axis = 0)
        else:
            ridge_normal_permute = np.mean(total_permute_ols_accs,axis=0)
    else:
        total_ds_accs = data
        cols = [1,2,0]
    ridge_normal, ridge_ttb, ridge_tally, _ = get_means(ds_num)
    b_ridge_accs=[];w_ridge_accs=[];
    b_ridge_ttb_accs=[];w_ridge_ttb_accs=[];
    b_ridge_tally_accs=[];w_ridge_tally_accs=[];
    b_ridge_permute_accs=[];w_ridge_permute_accs=[];
    all_ridge_accs=[];all_ridge_ttb_accs=[];all_ridge_tally_accs=[];
    lambda_ids=[]
    for i in range(niters_total): #0 normal, 1 ttb, 2 tal #.argmax()
        b_ridge_accs.append(total_ds_accs[i][ds_num][0][ridge_normal.argmax(),cols[0]])
        w_ridge_accs.append(total_ds_accs[i][ds_num][0][ridge_normal.argmin(),cols[0]])
        all_ridge_accs.append(total_ds_accs[i][ds_num][0][:,cols[0]])
        b_ridge_ttb_accs.append(total_ds_accs[i][ds_num][0][ridge_ttb.argmax(),cols[1]])
        w_ridge_ttb_accs.append(total_ds_accs[i][ds_num][0][ridge_ttb.argmin(),cols[1]])
        all_ridge_ttb_accs.append(total_ds_accs[i][ds_num][0][:,cols[1]])
        b_ridge_tally_accs.append(total_ds_accs[i][ds_num][0][ridge_tally.argmax(),cols[2]])
        w_ridge_tally_accs.append(total_ds_accs[i][ds_num][0][ridge_tally.argmin(),cols[2]])
        all_ridge_tally_accs.append(total_ds_accs[i][ds_num][0][:,cols[2]])
        #print(total_ds_accs[i][ds_num][0][:,cols[2]])
        lambda_ids.append(np.arange(len(lambda_list)))
        if accs:
            b_ridge_permute_accs.append(total_permute_ols_accs[i][ds_num][ridge_normal_permute.argmax()])
            w_ridge_permute_accs.append(total_permute_ols_accs[i][ds_num][ridge_normal_permute.argmin()])
    return b_ridge_accs, w_ridge_accs, b_ridge_ttb_accs, w_ridge_ttb_accs, b_ridge_tally_accs, w_ridge_tally_accs, b_ridge_permute_accs, w_ridge_permute_accs, \
    np.array(all_ridge_accs).flatten(), np.array(all_ridge_ttb_accs).flatten(), np.array(all_ridge_tally_accs).flatten(), lambda_ids

def get_plot2_violin(all_ds=0, accs=1):
    if accs:
        data = total_ds_accs, total_permute_ols_accs
    else:
        data = total_ds_entropies
    import seaborn as sns
    vector_size = np.size(total_tally_accs)
    if all_ds:
        b_ridge_accs=[];w_ridge_accs=[];
        b_ridge_ttb_accs=[];w_ridge_ttb_accs=[];
        b_ridge_tally_accs=[];w_ridge_tally_accs=[];
        b_ridge_permute_accs=[];w_ridge_permute_accs=[];
        ds_num_vec=[]; ds_num_vec2=[]
        all_ridge_accs=[]; all_ridge_ttb_accs=[]; all_ridge_tally_accs=[]
        for ds_num in range(len(ds_fn_list)):
            b_ridge_accs0, w_ridge_accs0, b_ridge_ttb_accs0, w_ridge_ttb_accs0, b_ridge_tally_accs0, \
            w_ridge_tally_accs0, b_ridge_permute_accs0, w_ridge_permute_accs0, \
            all_ridge_accs0, all_ridge_ttb_accs0, all_ridge_tally_accs0, lambda_ids = get_worst_best(data, ds_num, accs=accs)
            b_ridge_accs.append(b_ridge_accs0); w_ridge_accs.append(w_ridge_accs0);
            b_ridge_ttb_accs.append(b_ridge_ttb_accs0); w_ridge_ttb_accs.append(w_ridge_ttb_accs0);
            b_ridge_tally_accs.append(b_ridge_tally_accs0); w_ridge_tally_accs.append(w_ridge_tally_accs0);
            b_ridge_permute_accs.append(b_ridge_permute_accs0); w_ridge_permute_accs.append(w_ridge_permute_accs0);
            ds_num_vec.append(np.tile(ds_num,len(b_ridge_accs0)))
            all_ridge_accs.append(all_ridge_accs0)
            all_ridge_ttb_accs.append(all_ridge_ttb_accs0)
            all_ridge_tally_accs.append(all_ridge_tally_accs0)
            ds_num_vec2.append(np.tile(ds_num,len(all_ridge_accs0)))
        b_ridge_accs = np.array(b_ridge_accs).flatten(); w_ridge_accs = np.array(w_ridge_accs).flatten()
        b_ridge_ttb_accs = np.array(b_ridge_ttb_accs).flatten(); w_ridge_ttb_accs = np.array(w_ridge_ttb_accs).flatten()
        b_ridge_tally_accs = np.array(b_ridge_tally_accs).flatten(); w_ridge_tally_accs = np.array(w_ridge_tally_accs).flatten()
        b_ridge_permute_accs = np.array(b_ridge_permute_accs).flatten(); w_ridge_permute_accs = np.array(w_ridge_permute_accs).flatten()
        all_ridge_accs = np.array(all_ridge_accs).flatten()
        all_ridge_ttb_accs = np.array(all_ridge_ttb_accs).flatten()
        all_ridge_tally_accs = np.array(all_ridge_tally_accs).flatten()
    else:
        ds_num = 0
        b_ridge_accs, w_ridge_accs, b_ridge_ttb_accs, w_ridge_ttb_accs, b_ridge_tally_accs, \
        w_ridge_tally_accs, b_ridge_permute_accs, w_ridge_permute_accs, \
        all_ridge_accs, all_ridge_ttb_accs, all_ridge_tally_accs, lambda_ids = get_worst_best(data, ds_num, accs=accs)
        lambda_ids = np.array(lambda_ids).flatten()
    b_ridge_normal_data = np.hstack([np.tile('Best \n Zero Prior',(vector_size,1)),np.array([b_ridge_accs]).T])
    w_ridge_normal_data = np.hstack([np.tile('Worst \n Zero Prior',(vector_size,1)),np.array([w_ridge_accs]).T])
    b_ridge_ttb_data = np.hstack([np.tile('Best \n TTB Prior',(vector_size,1)),np.array([b_ridge_ttb_accs]).T])
    w_ridge_ttb_data = np.hstack([np.tile('Worst \n TTB Prior',(vector_size,1)),np.array([w_ridge_ttb_accs]).T])
    b_ridge_tally_data = np.hstack([np.tile('Best \n TAL Prior',(vector_size,1)),np.array([b_ridge_tally_accs]).T])
    w_ridge_tally_data = np.hstack([np.tile('Worst \n TAL Prior',(vector_size,1)),np.array([w_ridge_tally_accs]).T])
    vector_size2 = np.size(all_ridge_accs)
    all_ridge_accs_data = np.hstack([np.tile('Mean \n Zero Prior',(vector_size2,1)),np.array([all_ridge_accs]).T])
    all_ridge_ttb_accs_data = np.hstack([np.tile('Mean \n TTB Prior',(vector_size2,1)),np.array([all_ridge_ttb_accs]).T])
    all_ridge_tally_accs_data = np.hstack([np.tile('Mean \n TAL Prior',(vector_size2,1)),np.array([all_ridge_tally_accs]).T])
    if accs:
        b_ridge_normal_permute_data = np.hstack([np.tile('Best OLS \n Permuted \n Prior',(vector_size,1)),np.array([b_ridge_permute_accs]).T])
        w_ridge_normal_permute_data = np.hstack([np.tile('Worst OLS \n Permuted \n Prior',(vector_size,1)),np.array([w_ridge_permute_accs]).T])
        include_baselines = 0
        if include_baselines:
            tal_data = np.hstack([np.tile('TAL',(np.size(total_tally_accs),1)),np.array([np.array(total_tally_accs).flatten()]).T])
            ttb_data = np.hstack([np.tile('TTB',(np.size(total_ttb_accs),1)),np.array([np.array(total_ttb_accs).flatten()]).T])
            ols_data = np.hstack([np.tile('OLS',(np.size(total_ols_accs),1)),np.array([np.array(total_ols_accs).flatten()]).T])
            data = np.vstack([ols_data,tal_data,ttb_data,b_ridge_normal_data,w_ridge_normal_data,b_ridge_ttb_data,w_ridge_ttb_data,
            b_ridge_tally_data,w_ridge_tally_data,b_ridge_normal_permute_data,w_ridge_normal_permute_data])
            model2_labels = np.vstack([np.tile('0',(vector_size,1)),np.tile('1',(vector_size,1)),np.tile('2',(vector_size,1)),
            np.tile('3',(vector_size,1)),np.tile('3',(vector_size,1)),np.tile('4',(vector_size,1)),np.tile('4',(vector_size,1)),
            np.tile('5',(vector_size,1)),np.tile('5',(vector_size,1)),np.tile('6',(vector_size,1)),np.tile('6',(vector_size,1)) ])
        else:
            data = np.vstack([b_ridge_normal_data,w_ridge_normal_data,b_ridge_ttb_data,w_ridge_ttb_data,
            b_ridge_tally_data,w_ridge_tally_data,b_ridge_normal_permute_data,w_ridge_normal_permute_data,
            all_ridge_accs_data,all_ridge_ttb_accs_data,all_ridge_tally_accs_data])
            model2_labels = np.vstack([
            np.tile('b',(vector_size,1)),np.tile('b',(vector_size,1)),np.tile('g',(vector_size,1)),np.tile('g',(vector_size,1)),
            np.tile('r',(vector_size,1)),np.tile('r',(vector_size,1)),np.tile('y',(vector_size,1)),np.tile('y',(vector_size,1)),
            np.tile('b',(vector_size2,1)),np.tile('g',(vector_size2,1)),np.tile('r',(vector_size2,1))])
            xlabel_order = ['Best \n Zero Prior', 'Worst \n Zero Prior', 'Best \n TTB Prior', 'Worst \n TTB Prior',
            'Best \n TAL Prior', 'Worst \n TAL Prior', 'Best OLS \n Permuted \n Prior', 'Worst OLS \n Permuted \n Prior']#,
            #'Mean \n Zero Prior','Mean \n TTB Prior','Mean \n TAL Prior']
    else:
        data = np.vstack([b_ridge_normal_data,w_ridge_normal_data,b_ridge_ttb_data,w_ridge_ttb_data,
        b_ridge_tally_data,w_ridge_tally_data,all_ridge_accs_data,all_ridge_ttb_accs_data,all_ridge_tally_accs_data])
        model2_labels = np.vstack([
        np.tile('b',(vector_size,1)),np.tile('b',(vector_size,1)),np.tile('g',(vector_size,1)),np.tile('g',(vector_size,1)),
        np.tile('r',(vector_size,1)),np.tile('r',(vector_size,1)),
        np.tile('b',(vector_size2,1)),np.tile('g',(vector_size2,1)),np.tile('r',(vector_size2,1))])
        best = 0
        if best:
            xlabel_order = ['Best \n Zero Prior', 'Best \n TTB Prior','Best \n TAL Prior']
        else:
            xlabel_order = ['Mean \n Zero Prior', 'Mean \n TTB Prior','Mean \n TAL Prior']
    data = np.hstack([data,model2_labels])
    if accs:
        ylabel = 'Accuracy'
    else:
        ylabel = 'Normalized \n entropy'
    df = pd.DataFrame(data,columns=['Model',ylabel,'Model2'])
    df[ylabel] = pd.to_numeric(df[ylabel], downcast="float", errors='coerce')
    if all_ds:
        ds_num_vec3 = np.hstack([np.tile(np.array(ds_num_vec).flatten(),len(df.Model.unique())-3), np.tile(np.array(ds_num_vec2).flatten(),3)])
        df['ds_num'] = ds_num_vec3
        df2 = df.groupby(['ds_num','Model','Model2'], as_index=False).mean()
    else:
        if accs:
            df2 = df
        else:
            df['lambda_ids'] = np.hstack([np.tile(range(niters_total),len(df.Model.unique())-3), np.tile(lambda_ids,3)]) #idx for niter on best/worst models
            df2 = df.groupby(['lambda_ids', 'Model','Model2'], as_index=False).mean()
    plt.rcParams["figure.figsize"] = (20,10)
    fig, ax = plt.subplots()
    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()
    plt.ylim(ymax = 1, ymin = 0)
    f = sns.violinplot(x='Model', y=ylabel, data=df, color="0.8", zorder=1, ax=ax, order=xlabel_order) #hue="Model2",
    g = sns.stripplot(x='Model', y=ylabel, data=df2, jitter=True, zorder=1, hue="Model2", ax=ax,
    palette={'b':'b', 'g':'g', 'r':'r', 'y':'y', 'k':'k'}, alpha=0.5, order=xlabel_order)
    ax.legend_.remove()
    plt.rcParams.update({'font.size': 18})
    plt.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.16, wspace=0.265)
    plt.tight_layout()
    plt.show()

#get_plot2_violin(all_ds=1, accs=0)
#get_plot2_violin(all_ds=0, accs=0)

#get_plot2()
#plt.close()

if split_train:
    def get_var_plot():
        normalR_stds = []; normalR_mns = []
        ttbR_stds = []; ttbR_mns = []
        tallyR_stds = []; tallyR_mns = []
        zeroR_stds = []
        ols_stds = []
        ttb_stds = []
        tal_stds = []
        for i in range(len(ds_fn_list)):
            ridge_normal_std, ridge_ttb_std, ridge_tally_std, ridge_zero_std, big3_std = get_stds(i)
            normalR_stds.append(np.mean(ridge_normal_std))
            ttbR_stds.append(np.mean(ridge_ttb_std))
            tallyR_stds.append(np.mean(ridge_tally_std))
            zeroR_stds.append(np.mean(ridge_zero_std))
            ols_stds.append(big3_std[0])
            ttb_stds.append(big3_std[1])
            tal_stds.append(big3_std[2])
            ridge_normal_mns, ridge_ttb_mns, ridge_tally_mns, _, _ = get_means(i)
            normalR_mns.append(np.mean(ridge_normal_mns))
            ttbR_mns.append(np.mean(ridge_ttb_mns))
            tallyR_mns.append(np.mean(ridge_tally_mns))
        x_pos = np.arange(7)
        means = [np.mean(normalR_stds),np.mean(ttbR_stds),np.mean(tallyR_stds),np.mean(zeroR_stds),
            np.mean(ols_stds),np.mean(ttb_stds),np.mean(tal_stds)]
        stds = [np.std(normalR_stds),np.std(ttbR_stds),np.std(tallyR_stds),np.std(zeroR_stds),
            np.std(ols_stds),np.std(ttb_stds),np.std(tal_stds)]
        models = ['OLS prior', 'TTB prior', 'TAL prior', 'Zero prior', 'OLS', 'TTB', 'TAL']
        fig, ax = plt.subplots()
        bar_list = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
        bar_list[0].set_color('y'); bar_list[1].set_color('g'); bar_list[2].set_color('r');
        bar_list[3].set_color('b'); bar_list[4].set_color('y'); bar_list[5].set_color('g'); bar_list[6].set_color('r');
        ax.set_ylabel('Standard deviation of model accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.set_title(r'Model standard deviation (averaged over all datasets and $\theta$)')
        #ax.yaxis.grid(True)
        plt.show()
        from scipy import stats
        print('NormalR v TTBR: ', stats.ttest_rel(normalR_stds, ttbR_stds))
        print('NormalR v TALR: ', stats.ttest_rel(normalR_stds, tallyR_stds))
        print('NormalR v ZeroR: ', stats.ttest_rel(normalR_stds, zeroR_stds))
        print('OLS v TTB: ', stats.ttest_rel(ols_stds, ttb_stds))
        print('OLS v TAL: ', stats.ttest_rel(ols_stds, tal_stds))
        print('NormalR v TTBR (means): ', stats.ttest_rel(normalR_mns, ttbR_mns))
        print('NormalR v TALR (means): ', stats.ttest_rel(normalR_mns, tallyR_mns))
        print('NormalR (means w std): ', np.mean(normalR_mns), np.std(normalR_mns))
        print('TTBR (means w std): ', np.mean(ttbR_mns), np.std(ttbR_mns))
        print('TALR (means w std): ', np.mean(tallyR_mns), np.std(tallyR_mns))
        print('df: ', len(ttbR_mns)-1)

#ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(0)
#all_data = [all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs]
if testing==2:
    total_data = [total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements]
else:
    total_data = [total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements, total_permute_ols_accs, total_ds_entropies]

#np.save(parent_dir + 'breast_cancer_115n_1000iters',all_data)
#pickle.dump( total_data, open( parent_dir + sv_fn, "wb" ) )
#total_data_load = pickle.load( open( parent_dir + sv_fn, "rb" ) )

'''
f_TAL = pickle.load( open( '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/f_TAL', "rb" ) )
f_TTB = pickle.load( open( '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/f_TTB', "rb" ) )
f_Txx = f_TTB # f_TAL #
for funs in f_Txx:
    print(funs.fun, funs.x)
'''

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
