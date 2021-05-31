import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import numpy as np
import glob, os
import math
import datetime
import csv

from helper_functions import params


# Method to create sequence (batch- and neighborhood features) for a dataset
def get_sequence_features(df_data):
    
    # Read in A600 timestamp
    filename = '\extra_features\A600_Date.csv'
    df_A600 = ##path to data file
    
    # Join feature data with A600 timestamp to create an order(imput sequence order)
    df_data = df_data.join(df_A600)
    df_data.dropna(inplace=True)
    df_data.sort_values(by='Datetime_A600', inplace=True)
    df_data.drop(['Datetime_A600'], axis=1, inplace=True)

    # Perform a deep copy of the feature dataframe
    df_data_joined = df_data.copy()
    
    # Iterate over the preview period
    for i in range (0, params.preview_neighborhood):
        
        # Drop the first i indexes (depends on the preview cycle)
        df_data_shifted = df_data.drop(df_data.index[0:i+1])
        
        # Shift the dataframe by i
        df_data_shifted_renamed = df_data_shifted.rename(index=dict(zip(df_data_shifted.index,df_data.index[:-1])))
        
        # Join original data with the i-shifted data
        df_data_joined = df_data_joined.join(df_data_shifted_renamed, rsuffix=('_pre_' + str(i+1)))
    
    # Drop the original values from the dataframe (now it contains only the features of the i preceding vehicles)
    df_data_joined.dropna(inplace=True)
    df_data_joined.drop([col for col in df_data.columns if col in df_data_joined], axis=1, inplace=True)   
    
    # Perform a deep copy of the feature dataset
    df_data_rolling = df_data.copy()
    
    # Iterate over the columns
    for col in df_data.columns:
        # Create new rolling columns for each of the existing columns and fill them with the rolling mean of the preceding vehicles
        col_name = col + '_rolling_mean' + str(params.preview_batch_short)
        df_data_rolling[col_name] = df_data_rolling[col].rolling(preview_batch_short).sum()/preview_batch_short
        
        col_name = col + '_rolling_mean' + str(preview_batch_long)
        df_data_rolling[col_name] = df_data_rolling[col].rolling(preview_batch_long).sum()/preview_batch_long
        
    df_data_rolling.dropna(inplace=True)
    
    # Drop the original values from the dataframe (now it contains only the features of the i preceding vehicles) 
    df_data_rolling.drop([col for col in df_data.columns if col in df_data_rolling], axis=1, inplace=True)
    
    return df_data_joined, df_data_rolling


# Method to remove manually selected features from a dataframe
def remove_PrFams (df, remove_list):
    # Define columns not to delete
    cols = [c for c in df.columns if c[:5] not in remove_list]
    
    # Keep defined columns
    df_sample = df[cols]
    
    return df_sample
    

# Method to automatically store correleated feature columns in new, parental column
def feature_grouping(df_sample, save, save_bool):
    
    # Start with positive correlations
    # Calculate a correlation matrix 
    corrMatrix = df_sample.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    
    # Define a set to store the features that were already grouped
    already_in = set()
    positive_grouping_result = []
    
    # Iterate over the correlations matrix columns
    for col in corrMatrix:
        
        # Find a correlation of '1'
        perfect_corr = corrMatrix[col][corrMatrix[col] == 1].index.tolist()
        
        # If this feature is not considered yet, add it to the set and append it to the list of perfect correlations
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            positive_grouping_result.append(perfect_corr)
    
    # Create an empty dataframe and two empty lists   
    df_positive_group_features = pd.DataFrame()
    tuples = []
    features_in_positive_groups = []
    i = 0
    count = 0
    
    # Iterate over the list of positive correlations
    for g in positive_grouping_result:
        
        # Create a name for the new 'parental' features
        name = 'group' + str(i)
        
        # Assign value of the new feature to the original dataframe
        # The value equals the value of the first feature that is comprises in g (because all features are either '1' or '0')
        df_sample[name] = df_sample[g[0]]
        tuples.append([name, g])
        
        # Iterate over the features that are comprised in the new parental feature and drop them from the original dataframe
        for c in g:
            df_sample.drop(c, axis=1, inplace=True)
            features_in_positive_groups.append(c)
            count = count + 1
        i = i+1
        
    # Save the correlated tuples od parental feature and its respective name in a dictionary
    pos_group_dic = dict(tuples)
    
    df_positive_group_features = df_sample.iloc[:,-len(positive_grouping_result):]
    df_features_in_positive_groups = pd.DataFrame(features_in_positive_groups, columns=['features'])
    
    # Continue with the nefative correlations of '-1'
    # The prcodedure is exactly the same as above for the positive correlations
    corrMatrix = df_sample.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    
    already_in = set()
    negative_grouping_result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] == -1].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            negative_grouping_result.append(perfect_corr)
    
    df_negative_group_features = pd.DataFrame()
    tuples = []
    features_in_negative_groups = []
    
    i = 0
    count = 0
    for g in negative_grouping_result:
        name = 'neg_group' + str(i)
        df_sample[name] = df_sample[g[0]]
        tuples.append([name, g])
        for c in g:
            df_sample.drop(c, axis=1, inplace=True)
            features_in_negative_groups.append(c)
            count = count + 1
        i = i+1
        
    neg_group_dic = dict(tuples)
    
    df_negative_group_features = df_sample.iloc[:,-len(negative_grouping_result):]
    df_features_in_negative_groups = pd.DataFrame(features_in_negative_groups, columns=['features'])
    
    # Save Results if the parameter is set
    if (save_bool):
        
        filename = save + '\positive_groups.csv'
        df = pd.DataFrame(positive_grouping_result)
        df.to_csv(params.filepath_project_folder + filename)
    
        filename = save + '\positive_group_features.csv'
        df_positive_group_features.to_csv(filepath + filename)
    
        filename = save + '\positive_features_in_groups.csv'
        df_features_in_positive_groups.to_csv(filepath + filename)
    
        with open(save + '\positive_group_dic.csv', 'w', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in pos_group_dic.items():
                writer.writerow([key, value])
            
        filename = save + '\negative_groups.csv'
    
        df = pd.DataFrame(negative_grouping_result)
        df.to_csv(filepath + filename)
    
        filename = save + '\negative_group_features.csv'
        df_negative_group_features.to_csv(filepath + filename)
    
        filename = save + '\negative_features_in_groups.csv'
        df_features_in_negative_groups.to_csv(filepath + filename)
    
        with open(save + '\negative_group_dic.csv', 'w', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in neg_group_dic.items():
                writer.writerow([key, value])      
    
    # Return the dataframe containing the grouped features
    return df_sample
    
    
def percentage_cut_off (low, high, df):
    stats = df.describe()
    
    idx_high = stats.loc['mean'] > high
    idx_low = stats.loc['mean'] < low

    idx = idx_high + idx_low
    
    df_reduced = df.iloc[:, ~idx.values]
    
    return df_reduced

    
# Method to remove RS models froma dataframe    
def remove_RS(df):
    
    # Load the csv file containing the Kennnummern of RS models
    df_RS_KNr = pd.read_csv(params.filepath_project_folder + '\RS_KNr.csv', index_col=0)
    
    # Isolate the Kennnummern
    RS_KNr = df_RS_KNr['Kennnummer'].values
    
    # Remove those Kennnummern from the dataframe before returning it
    df_no_RS = df.loc[~df.index.isin(RS_KNr)]
    
    return df_no_RS
    

# Method to isolate the RS models in a dataset    
def keep_RS(df):
    
    # Load the csv file containing the Kennnummern of RS models
    df_RS_KNr = pd.read_csv(params.filepath_project_folder + '\RS_KNr.csv', index_col=0)
    
    # Isolate the Kennnummern
    RS_KNr = df_RS_KNr['Kennnummer'].values
    
    # Keep only the RS Kennnummern in the dataframe before returning it
    df_RS = df.loc[df.index.isin(RS_KNr)]
    
    return df_RS
    
# Method to remove audit models from a dataframe
# Works according to the ones above
def remove_audit(df):
    df_audits = pd.read_excel(params.filepath_project_folder + '\Audits.xlsx')
    audit_KNr = df_audits['Zusammenbau [ohne Trennzeichen]'].values
    df_no_audit = df.loc[~df.index.isin(audit_KNr)]
    
    return df_no_audit
    

# Method to keep only audit models in a dataframe
# Works according to the ones above 
def keep_audit(df):
    df_audits = pd.read_excel(params.filepath_project_folder + '\Audits.xlsx')
    audit_KNr = df_audits['Zusammenbau [ohne Trennzeichen]'].values
    df_audit = df.loc[df.index.isin(audit_KNr)]
    
    return df_audit
    

# Method to remove Messkarossen from a dataframe
# Works according to the ones above    
def remove_mess(df):
    df_mess_KNr = pd.read_csv(params.filepath_project_folder + '\Mess_KNr.csv', index_col=0)
    mess_KNr = df_mess_KNr['Kennnummer'].values
    df_no_mess = df.loc[~df.index.isin(mess_KNr)]
    
    return df_no_mess
    
# Method to keep only Messkarossen in a dataframe
# Works according to the ones above    
def keep_mess(df):
    df_mess_KNr = pd.read_csv(params.filepath_project_folder + '\Mess_KNr.csv', index_col=0)
    mess_KNr = df_mess_KNr['Kennnummer'].values
    df_mess = df.loc[df.index.isin(mess_KNr)]
    
    return df_mess    
