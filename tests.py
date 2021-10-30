
import numpy as np
import pandas as pd
import scipy.stats


def ks_test(train, test):
    rows, cols = train.shape
    d = np.zeros((cols, 1))
    pval = np.zeros((cols, 1))
    '''ks test'''
    for i in range(cols):
        d[i], pval[i] = stats.ks_2samp(train.iloc[:, i], test.iloc[:, i])
    if pval.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval
    
def kl_test(train, test):
    rows, cols = train.shape
    KL = np.zeros((cols, 1))
    total_time = 10*int(len(train) / len(test))
    eta = 1e-6
    for time in range(total_time):
        subsample = np.random.randint(0, len(train), (len(test), ))
        for i in range(cols):
            KL[i] += scipy.stats.entropy(train.iloc[subsample, i], test.iloc[:, i]+eta)
    KL /= total_time
    if KL.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return KL

def ftest(train,test):
    '''F检验样本总体方差是否相等'''
    F = np.var(train)/np.var(test)
    v1 = len(train) - 1
    v2 = len(test) - 1
    p_val = 1 - 2*abs(0.5-f.cdf(F,v1,v2))
    if p_val.all() < 0.05:
        print("Reject the Null Hypothesis.")
        equal_var=False
    else:
        print("Accept the Null Hypothesis.")
        equal_var=True
    return equal_var
    
def ttest_ind_fun(train,test):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(train,test)
    ttest,pval = ttest_ind(train,test,equal_var=equal_var)
    if pval.all() < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval



