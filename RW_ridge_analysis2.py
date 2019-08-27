import numpy as np
import os
import pandas as pd
import random
from scipy.optimize import minimize_scalar
from scipy import stats
import matplotlib.pyplot as plt
import time
import cProfile
import pickle

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

def run_ridges(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior,tally_prior):
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
    #B_ridge_normal = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i) + lambda_mat),np.dot(Xt_i,Y_train))
    #B_ridge_ttb = np.dot(np.linalg.pinv(np.dot(Xt_i_ttb,X_i_ttb) + lambda_mat),(np.dot(Xt_i_ttb,Y_train) + lam_bda*ttb_prior))
    #B_ridge_tally = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i) + lambda_mat),(np.dot(Xt_i,Y_train) + lam_bda*tally_prior))
    B_logR_tally, log_tally_acc, log_tally_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,tally_prior)
    B_logR_normal, log_normal_acc, log_normal_preds = log_regress_p(X_i,X_test,Y_train,Y_test,lam_bda,np.zeros((len(ttb_prior),)))
    B_logR_ttb, log_ttb_acc, log_ttb_preds = log_regress_p(X_i_ttb,X_test_ttb,Y_train,Y_test,lam_bda,ttb_prior)
    all_preds = [log_tally_preds, log_normal_preds, log_ttb_preds]
    #print(lam_bda,log_tally_acc,log_normal_acc,log_ttb_acc)
    #exit()
    #print(B_logR_normal,B_logR_ttb,B_logR_tally)
    #Y_normalR_preds = np.sign(np.dot(X_test,B_ridge_normal))
    #Y_ttbR_preds = np.sign(np.dot(X_test_ttb,B_ridge_ttb))
    #Y_tallyR_preds = np.sign(np.dot(X_test,B_ridge_tally))
    #normalR_acc = get_acc(Y_test,Y_normalR_preds)
    #ttbR_acc = get_acc(Y_test,Y_ttbR_preds)
    #tallyR_acc = get_acc(Y_test,Y_tallyR_preds)
    return log_normal_acc, log_ttb_acc, log_tally_acc, all_preds

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
    while diff_betas>0.001 and iters!=1000:
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
            print('big betas, perfect separation probable')
            break
    print('ginv2')
    pred_probs, _ = g_inv(X_test,beta_new)
    #print('ginv3')
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
    while diff_betas>0.001 and iters!=1000:
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
            print('big betas, perfect separation probable')
            break
    print('ginv_b')
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
    return acc, preds

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

rnd_seed = 666 #101
np.random.seed(rnd_seed)
random.seed(rnd_seed)
testing = -1 #-1 #1,-1,0,2
remove_zeros = 0
v = 0
niters_total = 1000
sample_size = 10
sigma = 1.3
#lambda_list = np.linspace(0.0001,0.1,num=50) #1000000000000 to converge
#lambda_list = np.hstack([0, np.geomspace(1,1000,num=50)]) #1000000000000 = 1e+12 to converge
lambda_list = np.hstack([0, np.linspace(0.00001,0.1,num=25), np.linspace(0.1,1000,num=20), np.geomspace(1000,1000000,num=5), np.geomspace(1000000,1000000000000,num=5)])
parent_dir = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/'
#parent_dir = '/home/seb/Dropbox/postdoc/LSS_project/20_classic_datasets/'
#parent_dir = os.getcwd() + '/'
data_dir = parent_dir + 'data/'
tmp_pair_data_dir = parent_dir + 'tmp_pair_data/'
#ds_fn_list = os.listdir(data_dir)
all_ttb_inds = np.load(tmp_pair_data_dir + 'ttb_inds.npy') #in same order as data_dir when loaded through listdir

ml_case = 'breast-cancer-wisconsin.data'
ml_case_labels = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion ','Single Epithelial Cell Size',
'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

if testing==1:
    test_fn = 'test_case_red2.csv'
    #test_fn = 'test_case_red.csv'
    #test_fn = 'test_case.csv'
    ds_fn_list = np.tile(test_fn,20)
elif testing==2:
    test_fn = 'not_testing_mode'
    ds_fn_list = ['simulation']
    sv_fn = 'simulation_190' + 'samples_' + str(niters_total) + 'iters'
elif testing == -1:
    test_fn = 'not_testing_mode'
    ds_fn_list = [ml_case]
    sv_fn = 'breast_cancer_' + str(sample_size) + 'samples_' + str(niters_total) + 'iters'
else:
    test_fn = 'not_testing_mode'
    sv_fn = '20ds_' + str(sample_size) + 'samples_' + str(niters_total) + 'iters'
    ds_fn_list = ['prf.world.txt', 'fish.fertility.txt', 'fuel.world.txt', 'attractiveness.men.txt', 'landrent.world.txt', 'dropout.txt',
    'attractiveness.women.txt', 'cloud.txt', 'car.world.txt', 'mortality.txt', 'bodyfat.world.txt', 'homeless.world.txt', 'oxygen.txt',
    'ozone.txt', 'mammal.world.txt', 'fat.world.txt', 'glps.txt', 'oxidants.txt', 'cit.world.txt', 'house.world.txt']

'''
paired_ds_list = ['pair_attractiveness.men.txt', 'pair_dropout.txt', 'pair_cloud.txt',
 'pair_bodyfat.world.txt', 'pair_landrent.world.txt', 'pair_car.world.txt', 'pair_mammal.world.txt',
 'pair_cit.world.txt', 'pair_oxidants.txt', 'pair_attractiveness.women.txt', 'pair_oxygen.txt',
 'pair_ozone.txt', 'pair_mortality.txt', 'pair_house.world.txt', 'pair_homeless.world.txt', 'pair_glps.txt',
 'pair_prf.world.txt', 'pair_fuel.world.txt', 'pair_fish.fertility.txt', 'pair_fat.world.txt']
'''

beta_start = 0

def main(niter, ds_fn_list,v=0):
    print('iteration: ' + str(niter+1) + ' of ' + str(niters_total))
    all_ds_accs=[]
    all_ds_agreements=[]
    all_ols_accs=[]
    all_ttb_accs=[]
    all_tally_accs=[]
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
            #print(tmp0); exit()
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
            for col in range(1,len(cols)):
                tmp_x = np.array(tmp0.iloc[:,col], dtype=int)
                tmp_median = np.median(tmp_x)
                tmp_x[tmp_x<tmp_median] = -1
                tmp_x[tmp_x==tmp_median] = 0
                tmp_x[tmp_x>tmp_median] = 1
                #tmp_x = stats.zscore(tmp_x)
                tmp0.iloc[:,col] = tmp_x
        elif ds_fn_list[ds_num] == 'simulation':
            Y_test, Y_train, X_test, X_train = sim_data()
        else:
            tmp0 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
            #if ds_num!=1:
                #tmp0 = pd.DataFrame(np.tile(tmp0, (3,1)))
        if ds_fn_list[ds_num] != 'simulation':
            tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
            if ds_fn_list[ds_num] == ml_case:
                tmp0_dels = np.sum(tmp0.iloc[:,0]==1) - np.sum(tmp0.iloc[:,0]==-1)
                tmp0_bool = tmp0.iloc[:,0]==1
                tmp0_bool_inds = np.where(tmp0_bool)
                tmp0_bool_inds = tmp0_bool_inds[0][0:tmp0_dels]
                tmp0.drop(tmp0.index[list(tmp0_bool_inds)], inplace=True)
        #print(tmp0.mean(axis=0))#;exit()
        #mid_ind = int(tmp0.shape[0]*0.5)
        #print(mid_ind)
        mid_ind=sample_size
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
            _, tally_prior = tallying(train_set,np.array([])) #if prior not calculated yet, send empty array
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
            _,ttb_inds,qvalis,flippers = compute_qvalis(train_set, flip_direction=1)
            ttb_preds, _ = ttbing(test_set,ttb_inds,qvalis,flippers)
            #print(np.sum(ttb_preds==0)); exit()
            ttb_acc = get_acc(Y_test,ttb_preds)
            ttb_accs.append(ttb_acc)
            ttb_prior0 = compute_ttb_prior0(ttb_inds,flippers)
            ttb_prior0 = ttb_prior0.reshape((ttb_prior0.shape + (1,)))
            ttb_prior0_train = np.tile(ttb_prior0,X_train.shape[0]).T
            ttb_prior0_test = np.tile(ttb_prior0,X_test.shape[0]).T
            X_train_ttb = (ttb_prior0_train*X_train)/np.max(np.abs(ttb_prior0_train))
            X_test_ttb = (ttb_prior0_test*X_test)/np.max(np.abs(ttb_prior0_train))
            f = minimize_scalar(lambda B: scaling_prior_log(Y_train, X_train_ttb, B),
            method='golden', options={'maxiter': 10000})
            ttb_prior = np.tile(f.x,X_train.shape[1]) #after ttbing X, we need a scaling prior
            #if ds_fn_list[ds_num] == test_fn:
                #ttb_prior = np.array([4,2,-1]) #use only when testing!
            #print(ttb_prior, tally_prior); exit()
            #ols_acc = regress(X_train,Y_train,X_test,Y_test)
            log_ols_acc, log_ols_preds = log_regress(X_train,X_test,Y_train,Y_test)
            ols_accs.append(log_ols_acc)
            cv_accs=[]
            cv_agreement = []
            for lam_bda in lambda_list:
                normalR_acc, ttbR_acc, tallyR_acc, preds3 = run_ridges(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior,tally_prior)
                tallyR_preds, normalR_preds, ttbR_preds = preds3
                cv_accs.append([normalR_acc, ttbR_acc, tallyR_acc])
                agree1 = np.array([get_acc(log_ols_preds,tallyR_preds), get_acc(tally_preds,tallyR_preds), get_acc(log_ols_preds,ttbR_preds), get_acc(ttb_preds,ttbR_preds)])
                agree2 = np.array([get_acc(tallyR_preds,log_ols_preds), get_acc(tallyR_preds,tally_preds), get_acc(ttbR_preds,log_ols_preds), get_acc(ttbR_preds,ttb_preds)])
                cv_agreement.append((agree1+agree2)/2)
            #exit()
            all_accs.append(cv_accs)
            all_agreements.append(cv_agreement)
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
        all_ds_accs.append(np.array(all_accs))
        all_ds_agreements.append(np.array(all_agreements))
        #exit()
    return all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs, all_ds_agreements

def parallelizer(niters_total,ds_fn_list,v):
    total_ols_accs=[]
    total_ttb_accs=[]
    total_tally_accs=[]
    total_ds_accs=[]
    total_ds_agreements=[]
    for niter in range(niters_total):
        all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs, all_ds_agreements = main(niter, ds_fn_list, v=v)
        total_ols_accs.append(all_ols_accs)
        total_ttb_accs.append(all_ttb_accs)
        total_tally_accs.append(all_tally_accs)
        total_ds_accs.append(all_ds_accs)
        total_ds_agreements.append(all_ds_agreements)
    return total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements

#cProfile.run('parallelizer(niters_total,ds_fn_list,v)')
total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements = parallelizer(niters_total,ds_fn_list,v)

all_ols_accs = np.mean(total_ols_accs,axis=0)
all_ttb_accs = np.mean(total_ttb_accs,axis=0)
all_tally_accs = np.mean(total_tally_accs,axis=0)
all_ds_accs = np.mean(total_ds_accs,axis=0)

def get_means(ds_num):
    big3 = [np.mean(all_ols_accs[ds_num]), np.mean(all_ttb_accs[ds_num]), np.mean(all_tally_accs[ds_num])]
    lambda_accs = np.mean(all_ds_accs[ds_num], axis=0)
    ridge_normal = lambda_accs[:,0]
    ridge_ttb = lambda_accs[:,1]
    ridge_tally = lambda_accs[:,2]
    return ridge_normal, ridge_ttb, ridge_tally, big3

def get_plot():
    if testing == -1:
        fig, ax = plt.subplots(nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(nrows=5, ncols=4)
    i = 0
    for row in ax:
        for col in row:
            ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(i)
            #y = np.hstack([big3, ridge_betas])
            col.plot(0, big3[0], 'bo') #ols
            col.plot(1, big3[1], 'g*') #ttb
            col.plot(2, big3[2], 'r+') #tally
            col.plot(range(len(big3), len(big3) + len(ridge_normal)), ridge_normal, 'bo')
            col.plot(range(len(big3), len(big3) + len(ridge_normal)), ridge_ttb, '*g')
            col.plot(range(len(big3), len(big3) + len(ridge_normal)), ridge_tally, '-r')
            col.set_title(ds_fn_list[i])
            if testing != -1:
                i += 1
    plt.subplots_adjust( hspace=0.36, wspace=0.2 , top=0.94 , left=0.07 , right=0.96 , bottom=0.03 )
    plt.show()

#get_plot()
#plt.close()


def get_plot2():
    ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(0)
    plt.rcParams["figure.figsize"] = (20,10)
    fig, ax = plt.subplots()
    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()
    ax.plot('OLS', big3[0], 'bo', label='OLS') #ols
    ax.plot('TTB', big3[1]-0.001, '*g', label='TTB') #ttb
    ax.plot('TAL', big3[2], 'r+', label='TAL') #tally
    ax.plot(lambda_list2[3:], ridge_normal, '-b', label="Zero prior")
    ax.plot(lambda_list2[3:], ridge_ttb, '-g', label="TTB prior")
    ax.plot(lambda_list2[3:], ridge_tally, '-r', label="TAL prior")
    ax.set_title("Breast Cancer Dataset")
    #ax.set_xticklabels(lambda_list2, rotation=45)
    plt.xticks(rotation=60)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.xlabel('Model/Penalty parameter', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    #plt.figure(num=1, figsize=(20,20), dpi=300, facecolor='w', edgecolor='k')
    fig.savefig(parent_dir + 'breast_cancer_data.png', bbox_inches='tight', dpi=300)
    plt.show()

lambda_list2=['OLS','TTB','TAL']
for l in lambda_list:
    lambda_list2.append('{:.1e}'.format(l))

#get_plot2()
#plt.close()

#ridge_normal, ridge_ttb, ridge_tally, big3 = get_means(0)
#all_data = [all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs]
if testing==2:
    total_data = [total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs, total_ds_agreements]
else:
    total_data = [total_ols_accs, total_ttb_accs, total_tally_accs, total_ds_accs]
#np.save(parent_dir + 'breast_cancer_115n_1000iters',all_data)
pickle.dump( total_data, open( parent_dir + sv_fn, "wb" ) )
#total_data_load = pickle.load( open( parent_dir + 'breast_cancer_115n_1000iters', "rb" ) )
