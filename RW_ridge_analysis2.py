import numpy as np
import os
import pandas as pd
import random
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

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
    #print cue_valies
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
    #print qvalis
    ttb_inds = np.argsort(qvalis)+1
    ttb_inds = ttb_inds[::-1]
    return ds, ttb_inds, qvalis, flippers

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

def tallying(ds_tmp, tally_dirs):
    ds = ds_tmp.copy()
    if tally_dirs.size==0:
        chng_tdirs = q_validities(ds)<0.5
        print 'tally qvals: ', q_validities(ds)
        tally_dirs = np.ones((ds.shape[1]-1,))
        tally_dirs[chng_tdirs] = -1
        #print chng_tdirs
    else:
        chng_tdirs = (tally_dirs*-1)+1
        #print tally_dirs, chng_tdirs; exit()
    #print ds.iloc[:,1:]
    for col in range(1,ds.shape[1]):
        if chng_tdirs[col-1]:
            ds.iloc[:,col] = ds.iloc[:,col]*-1
    tally_up = np.sum(ds.iloc[:,1:],axis=1)
    #print ds.iloc[:,1:]; exit()
    f = minimize_scalar(lambda B: scaling_prior(ds.iloc[:,0], ds.iloc[:,1:], B),
     method='golden', options={'maxiter': 10000})
    scaling = f.x
    #print ds
    #print 'beta: ', scaling
    #scaling_prior2(ds.iloc[:,0], ds.iloc[:,1:], scaling)
    #print 'beta: ', 1
    #scaling_prior2(ds.iloc[:,0], ds.iloc[:,1:], 1); #exit()
    tally_dirs = tally_dirs*scaling
    #print ds.iloc[:,0]; exit()
    tally_up = replace_zeros(tally_up)
    tally_preds = np.sign(tally_up) #/np.abs(tally_up)
    return tally_preds, tally_dirs

def replace_zeros(tally_up):
    tally_up = np.array(tally_up)
    for i in range(len(tally_up)):
        #print tally_up[i]
        if tally_up[i]==0:
            coin_flip = random.choice([1,-1])
            tally_up[i] = coin_flip
    return tally_up

def scaling_prior(y, x, B):
    B_vec = np.tile(B,x.shape[1])
    #y_pred = np.sign(np.dot(x,B_vec))
    y_pred = np.dot(x,B_vec)
    SSE = np.sum(np.power(y-y_pred,2))
    #print SSE
    return SSE

def scaling_prior2(y, x, B):
    B_vec = np.tile(B,x.shape[1])
    #y_pred = np.sign(np.dot(x,B_vec))
    y_pred = np.dot(x,B_vec)
    SSE = np.sum(np.power(y-y_pred,2))
    print np.power(y-y_pred,2)
    return SSE

def get_acc(y,ypreds):
    rnd_preds = np.sum(ypreds == 0)
    if rnd_preds > 0:
        rnd_acc_premium = rnd_preds*0.5
    else:
        rnd_acc_premium = 0
    acc = (np.sum(y==ypreds) + rnd_acc_premium)  /float(len(y))
    return acc

def ttbing(ds_tmp, ttb_inds, qvalis, flippers):
    if qvalis.size==0:
        ds,ttb_inds,qvalis,flippers = compute_qvalis(ds, flip_direction=1)
    else:
        ds = ds_tmp.copy()
    #chngq = qvalis<0.5
    #for col in range(1,ds.shape[1]):
    #    if chngq[col-1]:
    #        ds.iloc[:,col] = ds.iloc[:,col]*-1
    ttb_preds = np.zeros((ds.shape[0],))
    for i in range(len(ttb_inds)):
        x_i = np.array(ds.iloc[:,ttb_inds[i]]*flippers[ttb_inds[i]-1])
        msk = ttb_preds!=0
        x_i[msk] = 0
        ttb_preds += x_i
    ttb_preds = replace_zeros(ttb_preds)
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
    B_ridge_normal = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i) + lambda_mat),np.dot(Xt_i,Y_train))
    B_ridge_ttb = np.dot(np.linalg.pinv(np.dot(Xt_i_ttb,X_i_ttb) + lambda_mat),(np.dot(Xt_i_ttb,Y_train) + lam_bda*ttb_prior))
    #B_ridge_ttb = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i) + lambda_mat),(np.dot(Xt_i,Y_train) + lam_bda*ttb_prior))
    B_ridge_tally = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i) + lambda_mat),(np.dot(Xt_i,Y_train) + lam_bda*tally_prior))
    Y_normalR_preds = np.sign(np.dot(X_test,B_ridge_normal))
    #Y_ttbR_preds = np.sign(np.dot(X_test_ttb,B_ridge_ttb))
    Y_ttbR_preds = np.sign(np.dot(X_test,B_ridge_ttb))
    Y_tallyR_preds = np.sign(np.dot(X_test,B_ridge_tally))
    #Y_normalR_preds = replace_zeros(Y_normalR_preds)
    #Y_tallyR_preds = replace_zeros(Y_tallyR_preds)
    #Y_ttbR_preds = replace_zeros(Y_ttbR_preds)
    #print B_ridge_normal
    #print B_ridge_ttb
    #print B_ridge_tally, B_ridge_normal; exit()
    #Y_tallyR_preds = np.sign(np.dot(X_i,tally_prior))
    normalR_acc = get_acc(Y_test,Y_normalR_preds)
    ttbR_acc = get_acc(Y_test,Y_ttbR_preds)
    tallyR_acc = get_acc(Y_test,Y_tallyR_preds)
    #print normalR_acc, tallyR_acc; exit()
    return normalR_acc, ttbR_acc, tallyR_acc

def regress(X_train,Y_train,X_test,Y_test):
    X_i = X_train
    Xt_i = np.transpose(X_i)
    B_OLS = np.dot(np.linalg.pinv(np.dot(Xt_i,X_i)),np.dot(Xt_i,Y_train))
    Y_OLS_preds = np.sign(np.dot(X_test,B_OLS))
    Y_OLS_preds = replace_zeros(Y_OLS_preds)
    #OLS_acc = np.sum(Y_OLS_preds==Y_test)/float(len(Y_OLS_preds))
    OLS_acc = get_acc(Y_test,Y_OLS_preds)
    return OLS_acc

rnd_seed = 666 #101
np.random.seed(rnd_seed)
random.seed(rnd_seed)
testing = 1
v = 0
#niters = 20 #average over variability in tallying and ttbing
niters_total = 1
lambda_list = np.linspace(0,100000,num=25)
#print lambda_list; exit()
#parent_dir = '/media/seb/HD_Numba_Juan/Dropbox/postdoc/LSS_project/20_classic_datasets/'
parent_dir = '/home/seb/Dropbox/postdoc/LSS_project/20_classic_datasets/'
#parent_dir = os.getcwd() + '/'
data_dir = parent_dir + 'data/'
tmp_pair_data_dir = parent_dir + 'tmp_pair_data/'
#ds_fn_list = os.listdir(data_dir)
all_ttb_inds = np.load(tmp_pair_data_dir + 'ttb_inds.npy') #in same order as data_dir when loaded through listdir

if testing==1:
    test_fn = 'test_case_red2.csv'
    #test_fn = 'test_case_red.csv'
    #test_fn = 'test_case.csv'
    ds_fn_list = np.tile(test_fn,20)
else:
    test_fn = 'not_testing_mode'
    ds_fn_list = ['prf.world.txt', 'fish.fertility.txt', 'fuel.world.txt', 'attractiveness.men.txt', 'landrent.world.txt', 'dropout.txt',
    'attractiveness.women.txt', 'cloud.txt', 'car.world.txt', 'mortality.txt', 'bodyfat.world.txt', 'homeless.world.txt', 'oxygen.txt',
    'ozone.txt', 'mammal.world.txt', 'fat.world.txt', 'glps.txt', 'oxidants.txt', 'cit.world.txt', 'house.world.txt']

'''
paired_ds_list = ['pair_attractiveness.men.txt', 'pair_dropout.txt', 'pair_cloud.txt',
 'pair_bodyfat.world.txt', 'pair_landrent.world.txt', 'pair_car.world.txt', 'pair_mammal.world.txt',
 'pair_cit.world.txt', 'pair_oxidants.txt', 'pair_attractiveness.women.txt', 'pair_oxygen.txt',
 'pair_ozone.txt', 'pair_mortality.txt', 'print ttb_priorpair_house.world.txt', 'pair_homeless.world.txt', 'pair_glps.txt',
 'pair_prf.world.txt', 'pair_fuel.world.txt', 'pair_fish.fertility.txt', 'pair_fat.world.txt']
'''

def main(ds_fn_list,v=0):
    print 'iteration: ' + str(iter+1) + ' of ' + str(niters_total)
    all_ds_accs=[]
    all_ols_accs=[]
    all_ttb_accs=[]
    all_tally_accs=[]
    for ds_num in range(len(ds_fn_list)):
        if v==1:
            print ds_fn_list[ds_num]
        if ds_fn_list[ds_num] == test_fn:
            tmp0 = pd.read_csv(parent_dir + test_fn, sep=",")
            #tmp0 = pd.DataFrame(np.tile(tmp0, (20,1)))
            #print tmp0; exit()
        else:
            tmp0 = pd.read_csv(tmp_pair_data_dir + 'pair_' + ds_fn_list[ds_num], sep=",")
        tmp0 = tmp0.reindex(np.random.permutation(tmp0.index))
        print tmp0.shape
        mid_ind = tmp0.shape[0]/2
        #mid_ind=5
        all_accs=[]
        tally_accs=[]
        ttb_accs=[]
        ols_accs=[]
        for cv in range(2):
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
            ols_acc = regress(X_train,Y_train,X_test,Y_test)
            ols_accs.append(ols_acc)
            _, tally_prior = tallying(train_set,np.array([])) #if prior not calculated yet, send empty array
            #if ds_fn_list[ds_num] == test_fn:
                #tally_prior = np.array([1,1,-1]) #use only when testing!
            #tally_acc=[]
            #for t in range(niters):
            tally_preds, _ = tallying(test_set,np.sign(tally_prior))
            #tally_acc_tmp = np.sum(tally_preds==Y_test)/float(len(tally_preds))
            tally_acc = get_acc(Y_test,tally_preds)
            #    tally_acc.append(tally_acc_tmp)
            #tally_acc = np.median(tally_acc)
            tally_accs.append(tally_acc)
            _,ttb_inds,qvalis,flippers = compute_qvalis(train_set, flip_direction=1)
            #ttb_acc=[]
            #for t in range(niters):
            ttb_preds, _ = ttbing(test_set,ttb_inds,qvalis,flippers)
            ttb_acc = get_acc(Y_test,ttb_preds)
            #    ttb_acc_tmp = np.sum(ttb_preds==Y_test)/float(len(ttb_preds))
            #    ttb_acc.append(ttb_acc_tmp)
            #ttb_acc = np.median(ttb_acc)
            ttb_accs.append(ttb_acc)
            ttb_prior0 = compute_ttb_prior0(ttb_inds,flippers)
            ttb_prior0 = ttb_prior0.reshape((ttb_prior0.shape + (1,)))
            ttb_prior0_train = np.tile(ttb_prior0,X_train.shape[0]).T
            ttb_prior0_test = np.tile(ttb_prior0,X_test.shape[0]).T
            X_train_ttb = ttb_prior0_train*X_train
            X_test_ttb = ttb_prior0_test*X_test
            f = minimize_scalar(lambda B: scaling_prior(Y_train, X_train_ttb, B),
            method='golden', options={'maxiter': 10000})
            ttb_prior = np.tile(f.x,X_train.shape[1]) #after ttbing X, we need a scaling prior
            #if ds_fn_list[ds_num] == test_fn:
                #ttb_prior = np.array([4,2,-1]) #use only when testing!
            #print ttb_prior, tally_prior; exit()
            cv_accs=[]
            for lam_bda in lambda_list:
                #normalR_acc=[]
                #ttbR_acc=[]
                #tallyR_acc=[]
                #for t in range(niters):
                normalR_acc, ttbR_acc, tallyR_acc = run_ridges(X_train,X_train_ttb,Y_train,X_test,X_test_ttb,Y_test,lam_bda,ttb_prior,tally_prior)
                #    normalR_acc.append(normalR_acc_tmp)
                #    ttbR_acc.append(ttbR_acc_tmp)
                #    tallyR_acc.append(tallyR_acc_tmp)
                #cv_accs.append([np.mean(normalR_acc), np.mean(ttbR_acc), np.mean(tallyR_acc)])
                cv_accs.append([normalR_acc, ttbR_acc, tallyR_acc])
            all_accs.append(cv_accs)
            #print 'tally_prior: ', tally_prior
            #exit()
            if v==1:
                print 'tally: ', np.mean(tally_accs)
                print 'tally_prior: ', tally_prior
                print 'ttb: ', np.mean(ttb_accs)
                print 'ttb_prior: ', ttb_prior
                print 'ols: ', np.mean(ols_accs)
                all_accs_means = np.mean(all_accs, axis=0)
                print 'max ridge (normal, ttb_prior, tally_prior):', np.max(all_accs_means,axis=0)
                print '--------------------------------------------------------------------'
        all_ols_accs.append(np.array(ols_accs))
        all_ttb_accs.append(np.array(ttb_accs))
        all_tally_accs.append(np.array(tally_accs))
        all_ds_accs.append(np.array(all_accs))
    return all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs

total_ols_accs=[]
total_ttb_accs=[]
total_tally_accs=[]
total_ds_accs=[]
for iter in range(niters_total):
    all_ols_accs, all_ttb_accs, all_tally_accs, all_ds_accs = main(ds_fn_list, v=v)
    total_ols_accs.append(all_ols_accs)
    total_ttb_accs.append(all_ttb_accs)
    total_tally_accs.append(all_tally_accs)
    total_ds_accs.append(all_ds_accs)

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
            i += 1
    plt.subplots_adjust( hspace=0.36, wspace=0.2 , top=0.94 , left=0.07 , right=0.96 , bottom=0.03 )
    plt.show()


get_plot()
plt.close()
