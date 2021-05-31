import pandas as pd
import matplotlib.pyplot as plt
import re
import random
import numpy as np
import glob, os
import math
import datetime

from helper_functions import params

# Method do load all filtered and vectorized instances that are definied in param_tuples
# param_tuples is a list of tuples (TMA, count PR families)
def load_prepro_PNR(path):
    
    # Create a new, empty list
    dfs = []
    
    # Iterate over the definied parameter tuples
    for p in params.param_tuples_TMA_PRFam:
        print('# of PrFams: ' + str(p[1]))
        print('Modell: ' + str(p[0]))
        
        # Create new, empty, temporary list 
        dfs_temp = []
        
        # Create individual filenames to load according to the pattern and the definied parameter tuple
        filenames = glob.glob(path + '/_' + str(p[1]) + '_' + str(p[0]) + '*.csv')
        
        # Load all csv-files with that filename pattern and append them to the temporary list
        for filename in filenames:
            frame = pd.read_csv(filename)
            print('file: ' + filename + ': ' + str(len(frame)))
            dfs_temp.append(frame)
        
        # Concatenate the temporary list and append it to the final list as a dictionary items
        # The dict name is the TMA_number of PR families
        df_temp = pd.concat(dfs_temp, ignore_index=True, sort=False)
        df_temp.set_index(['Kennnummer'], drop=True, inplace=True)
        dic_name = str(p[0]) + '_' + str(p[1])
        dfs.append([dic_name, df_temp])
    
    # Formally transform the list intoa dictionary containing instances, grouped by tuples of TMA and number of PR families
    dic = dict(dfs)
    
    return dic
    
# Method to preprocess the dataframe
# This includes removal of duplicates and invalid values
# Remove vehicles that were not from Werk = 22 (Neckarsulm) 
def preprocessing (df_in):
    
    # Perform a deep copy of the dataframe
    df = df_in.copy()
    
    # Rename columns, if mispelled
    if (df.columns.contains("Unnamed: 0")):
        df.rename(columns={'Unnamed: 0':"Kennnummer"}, inplace=True)
        print('Sucessfully renamed column "Unnamed: 0" to "Kennnummer"')
    if (df.columns.contains("Kenn number")):
        df.rename(columns={'Kenn number':"Kennnummer"}, inplace=True)
        print('Sucessfully renamed column "Kenn number" to "Kennnummer"')
  
    # Count lenght before preprocessing
    rows_pre = len(df)
    
    # Iterate over columns
    # Remove invalid entries
    for column in df:
        df = df[~df[column].isin(['#MULTIVALUE'])]
    
    # Count length and print delta
    rows_post = len(df)
    delta_rows = rows_pre-rows_post
    print('Sucessfully removed ' + str(delta_rows) + ' rows from Dataframe due to #MULTIVALUE')
    
    # If index is not Kennnummer, then reset the index to Kennnummer
    if df.index.name != 'Kennnummer':
        df.set_index('Kennnummer', drop=True, inplace=True)
        print('Index was reset to "Kennnummer"')
    
    # Count length
    rows_pre = len(df)
    
    # Find indices of duplicates (all instances in case of duplicates)
    duplicates = df.index.duplicated(keep=False)
    
    # Store duplicates in separated dataframe
    df_deletes_dupl = df[duplicates]
    
    # Remove duplicates from original dataframe
    df = df.loc[~df.index.duplicated(keep=False)]
    
    # Count length and print delta
    rows_post = len(df)
    delta_rows = rows_pre-rows_post
    print('Sucessfully removed ' + str(delta_rows) + ' duplicated rows from Dataframe due to duplicated Kennnumer')
    
    # Count length
    rows_pre = len(df)
    
    # Keep only instances from Werk = 22
    df = df[df.index.str[:2] == '22']
    
    # Count length and print delta
    rows_post = len(df)
    delta_rows = rows_pre-rows_post
    print('Sucessfully removed ' + str(delta_rows) + ' rows from Werk != 22')
    
    # Remove commas from numerical values 
    for column in df:
        df[column].replace(',','',regex=True,inplace=True)
        df[column] = df[column].astype(int)
    print('Sucessfully removed seperators and converted to int')
    
    return df, df_deletes_dupl
    
# Method to add text labels to PR families    
def create_PrNrn_text (df_joined):
    
    # Load excel file with PrNrn texts, organized by PR families and numbers
    df_prntext = pd.read_excel(params.filepath_project_folder + '\PrNrn_Text.xlsx')
    
    # Extract rows with missing text information (those are the rows where PR numbers can be found)
    # Text is only in the rows that contain PR families
    df_nan = df_prntext.isnull()
    df_prntext['Beschreibung_Fam'] = ""
    
    # Iterate over the rows of the excel file
    for i in range (0, len(df_prntext)):
        
        # If a row with a new family is reached, save text of the new family
        if not df_nan.loc[i,'Fam']:
            fam = df_prntext.loc[i,'Fam']
            text = df_prntext.loc[i,'Beschreibung']
        # If no new family, then use the text of the current (last) family and add the text of the PR number
        if df_nan.loc[i,'Fam']:
            df_prntext.at[i,'Fam'] = fam
            df_prntext.at[i,'Beschreibung_Fam'] = text
    
    # Drop empty rows that might exist
    df_prntext.dropna(inplace=True)
    
    # Shape the label according to the desired pattern
    df_prntext['Fam_Num'] = '~1' + df_prntext['Fam'].astype(str) + '_~2' + df_prntext['Num'].astype(str)
    df_prntext['Beschreibung_joined'] = df_prntext['Beschreibung_Fam'].astype(str) + '_' + df_prntext['Beschreibung'].astype(str)
    
    # Set the label text as index
    df_prntext.set_index('Fam_Num', inplace = True)
    
    # Extract the column headers of the existing dataframe of PR numbers and join with the text labels
    df_cols = pd.DataFrame(df_joined.columns, columns=["PrNr"])
    df_cols = df_cols.join(df_prntext, on='PrNr')  
    df_cols['new_text'] = df_cols['PrNr'] + ' ' + df_cols['Beschreibung_joined']
    
    # Replace numerical text labels with PrNr information
    for i in range (0, len(df_cols)):
        if (type(df_cols.loc[i, 'new_text']) == float):
            if math.isnan(df_cols.loc[i, 'new_text']):
                df_cols.at[i, 'new_text'] = df_cols.loc[i, 'PrNr']
        
    cols = df_cols['new_text'].values
    
    # Temporarily remove Kennnummern and reset index, then exchange the column header
    df_joined.reset_index(drop=False, inplace=True)
    kennr = df_joined['Kennnummer']
    df_joined.drop(['Kennnummer'], axis=1, inplace=True)
    vals = df_joined.values
    df_joined_new = pd.DataFrame(vals, columns = cols)
    
    # Set back index to Kennnummer
    df_joined_new['Kennnummer'] = kennr
    df_joined_new.set_index(['Kennnummer'], drop=True, inplace=True)
        
    return(df_joined_new)
    

# Method to remove feature columns that are always filled with identical values
# i.e. a PR number is always present or never occuring
def remove_constant_prnrn (df_joined):
    
    # Count feature columns before removal
    features_pre = len(df_joined.columns)
    
    # Calculate column sums and store separately
    df_sum = pd.DataFrame(df_joined.sum(), columns=['Summe'])
    
    # Count length of dataset
    total = len(df_joined)
    
    # Find always present features by comparing the sum of the binary feature column with the total amount of rows
    # If sum == number of rows, the feature is present in every row
    df_unique = df_sum[df_sum['Summe'] == total]
    
    # If a sum for a feature column is zero, than this feature is never present 
    df_never = df_sum[df_sum['Summe'] == 0]
    
    # Extract indexes of always or never present features
    df_unique.reset_index(inplace=True)
    uniques = df_unique['index'].values
    df_never.reset_index(inplace=True)
    never = df_never['index'].values
    
    # Drop feature columns
    for col in uniques:
        df_joined.drop([col], axis=1, inplace=True)

    for col in never:
        df_joined.drop([col], axis=1, inplace=True)
    
    # Count feature columns after removal and print delta
    features_post = len(df_joined.columns)
    
    print('PrNrn that were always present (always = 1):')
    print(uniques)
    print()
    print('PrNrn that were never present (always = 0):')
    print(never)
    print()
    print('The number of features/columns reduces from ' + str(features_pre) + ' to ' + str(features_post))
    print()
    
    return (df_joined, uniques, never)
    

# Method to read data of Messungen and store Kennnummer of those vehicles    
def get_audit_vehicles():
    # Create temporary list and define filenames of vehicles that got measured
    dfs_temp = []
    filenames = glob.glob(params.filepath_project_folder + '\\database extracts\\audits' + '\\*' + '*.csv')
    # Iterate over all files containing audits and store in a dataframe
    for filename in filenames:
        frame = pd.read_csv(filename, sep=';')
        print('file: ' + filename + ': ' + str(len(frame)))
        dfs_temp.append(frame)
    df_temp = pd.concat(dfs_temp, ignore_index=True, sort=False)
    
    # Set validity of each instance to zero by default
    # Now we check if entries are valid and adjust them if necessary
    df_temp['valid'] = 0
    
    # Define pattern of a valid Kennnummer
    regexp = re.compile(r'22[0-9]{4}\d{7}')
    
    # Iterate over rows and extract the Kennnummer
    for index, row in df_temp.iterrows():
        string = df_temp.iloc[index]['K6 Identnummer']
        
        # If Kennnummer not in type string, then convert to string
        if type(string) != str:
            string = str(string)
        
        # If Kennnummer is valid, then turn validity to oen
        if regexp.search(string):
            df_temp.at[index, 'valid'] = 1
    
    # Copy valid instances into a separate dataframe, reset index and convert to string
    df_temp = df_temp[df_temp['valid'] == 1]
    df_temp.reset_index(inplace=True, drop=True)
    df_temp['K6 Identnummer'] = df_temp['K6 Identnummer'].astype(str)
    
    # Create a new column Kennnummer and bring the Kennnummer into the desired format 'XX-XXXX-XXXXXXX'
    df_temp['Kennnummer'] = ""
    for i in range (0, len(df_temp)):
        df_temp['Kennnummer'][i] = df_temp.iloc[i]['K6 Identnummer'][:2] + '-' + df_temp.iloc[i]['K6 Identnummer'][2:6] + '-' + df_temp.iloc[i]['K6 Identnummer'][6:13]
    
    # Store Kennnummern of measured vehicles in new dataframe, remove duplicates and save as csv
    df_temp = df_temp[['Kennnummer']]
    dupl = df_temp.duplicated(keep='first')
    df_temp = df_temp[~dupl]
    df_temp.reset_index(inplace=True, drop=True)
    
    df_temp.to_csv(params.filepath_project_folder + '\\Mess_KNr.csv')
    
    return df_temp
    
 