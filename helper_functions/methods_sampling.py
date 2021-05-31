import pandas as pd
import numpy as np
from time import time

from collections import Counter
from imblearn.over_sampling import (SMOTE, 
                                    ADASYN)

from imblearn.combine import SMOTEENN, SMOTETomek 

def sample_data(df_train, residual_col, sampling):

    #t0 = time()

    if (sampling == 'none'):
        train_balanced = df_train
    if (sampling == 'RandomUndersample'):
        train_balanced = randomUndersample(df_train, residual_col)
    if (sampling == 'RandomOversample'):
        train_balanced = randomOversample(df_train, residual_col)
    if (sampling == 'SMOTEOver'):
        train_balanced = SMOTEOversample(df_train, residual_col)
    if (sampling == 'ADASYN'):
        train_balanced = ADASYNOversample(df_train, residual_col)
    if (sampling == 'SMOTEENN'):
        train_balanced = SMOTEENNSampling(df_train, residual_col)
    if (sampling == 'SMOTETomek'):
        train_balanced = SMOTETomekSampling(df_train, residual_col)
    
    #t1 = time()
    #print()
    #print('Sampling took %f' %(t1-t0))
    #print()
    
    return train_balanced


def randomUndersample(df_train, residual_col):
    
    print()
    print('Perform RandomUndersample')
    n_classes = df_train[residual_col].nunique()
    n_instances = df_train[residual_col].value_counts().min() 
    
    df_balanced = pd.DataFrame(columns=df_train.columns)
    
    for i in range (0, n_classes):
        df_balanced = df_balanced.append(df_train[df_train[residual_col] == i].sample(n=n_instances, random_state=1))
    
    print ('New dataset contains ' + str(n_instances) + ' per class in ' + str(n_classes) + ' classes')

    return df_balanced


def randomOversample(df_train, residual_col):
    
    print()
    print('Perform RandomOversample')
    n_classes = df_train[residual_col].nunique()
    n_instances_max = df_train[residual_col].value_counts().max()
    
    majority_class_label = int(df_train[residual_col].value_counts().idxmax())
    print('Majority-Class Label: ' + str(majority_class_label))
    
    df_balanced = df_train
    
    for i in range (0, n_classes):
        n_instances = len(df_train[df_train[residual_col] == i])
        print('Label ' + str(i) + ' identified with ' + str(n_instances) + ' instances')
        if n_instances < n_instances_max:
            multiplier = int(n_instances_max/n_instances)
            print('Multiplier for the class is ' + str(multiplier))
            for j in range (0, multiplier):
                #print('Iteration ' + str(j) + '/' + str(multiplier))
                df_balanced = df_balanced.append(df_train[df_train[residual_col] == i])
            print('Dataframe now has ' + str(len(df_balanced)) + ' instances')
            
    print ('New dataset contains ' + str(n_instances_max) + ' per class in ' + str(n_classes) + ' classes')
    print ('In total: ' + str(len(df_balanced)))
    
    df_balanced = randomUndersample(df_balanced, residual_col)
    
    return df_balanced


def SMOTEOversample(df_train, residual_col):
    
    print()
    print('Perform SMOTEOversample')
    print('Original dataset shape %s' %Counter(df_train[residual_col].values))
    df_balanced = randomOversample(df_train, residual_col)
   
    sm = SMOTE(n_jobs=-1)
    
    X = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1:].values
    
    X_res, y_res = sm.fit_resample(X, y)
    
    df_balanced = pd.DataFrame(X_res, columns=[df_train.columns[:-1]])
    df_balanced['delta_A600_E100_cat'] = y_res
    
    print('Balanced dataset shape %s' %Counter(y_res))
    print('In total: ' + str(len(df_balanced)))
    
    return df_balanced
    

def ADASYNOversample(df_train, residual_col):
    
    print()
    print('Perform ADASYNOversample')
    print('Original dataset shape %s' %Counter(df_train[residual_col].values))
    ada = ADASYN(n_jobs=-1)
    
    X = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1:].values
    
    X_res, y_res = ada.fit_resample(X, y.ravel())
    
    df_balanced = pd.DataFrame(X_res, columns=[df_train.columns[:-1]])
    df_balanced['delta_A600_E100_cat'] = y_res.astype(int)
    
    print('Balanced dataset shape %s' %Counter(y_res))
    print('In total: ' + str(len(df_balanced)))
    
    return df_balanced


def SMOTEENNSampling(df_train, residual_col):
    
    print()
    print('Perform SMOTEENN')
    print('Original dataset shape %s' %Counter(df_train[residual_col].values))
    sme = SMOTEENN(n_jobs=-1)
    
    X = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1:].values
    
    X_res, y_res = sme.fit_resample(X, y.ravel())
    
    df_balanced = pd.DataFrame(X_res, columns=[df_train.columns[:-1]])
    df_balanced['delta_A600_E100_cat'] = y_res.astype(int)
    
    #type(df_balanced[residual_col][0])
    #df_balanced = df_balanced.astype(int)
    #type(df_balanced[residual_col][0])
    
    print('Balanced dataset shape %s' %Counter(y_res))
    print('In total: ' + str(len(df_balanced)))
    
    return df_balanced 


def SMOTETomekSampling(df_train, residual_col):
    
    print()
    print('Perform SMOTETomek')
    print('Original dataset shape %s' %Counter(df_train[residual_col].values))
    smt = SMOTETomek(n_jobs=-1)
    
    X = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1:].values
    
    X_res, y_res = smt.fit_resample(X, y.ravel())
    
    df_balanced = pd.DataFrame(X_res, columns=[df_train.columns[:-1]])
    df_balanced['delta_A600_E100_cat'] = y_res.astype(int)
    
    print('Balanced dataset shape %s' %Counter(y_res))
    print('In total: ' + str(len(df_balanced)))
    
    return df_balanced    
    
