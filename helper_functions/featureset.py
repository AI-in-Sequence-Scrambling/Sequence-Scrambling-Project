import pandas as pd

from helper_functions import feature_engineering

def provide_dataset(featureset_paramgrid):

    remove_RS = featureset_paramgrid['remove_RS']
    group = featureset_paramgrid['group']
    base_features = featureset_paramgrid['base_features']
    sequence = featureset_paramgrid['sequence']
    
    filepath = ##path to data file
    df_sqa = pd.read_csv(filepath + '\SQA_full_prepro_data.csv', index_col=0)

    if base_features == "all":
        print('Individual features used: All available')
        if group == True:
            print('Automated grouping: True')
            df_features = pd.read_csv(filepath + '\\features_full_reduced.csv', index_col=0)
        if group == False:
            print('Automated grouping: False')
            df_features = pd.read_csv(filepath + '\\features_full.csv', index_col=0)
    
    if base_features == "kaco_laco":
        print('Individual features used: Karossencode')
        if group == True:
            print('Automated grouping: True')
            df_features = pd.read_csv(filepath + '\\features_kaco_reduced.csv', index_col=0)
        if group == False:
            print('Automated grouping: False')
            df_features = pd.read_csv(filepath + '\\features_kaco.csv', index_col=0)
            
    if base_features == "manual":
        print('Individual features used: Manual Removal')
        if group == True:
            print('Automated grouping: True')
            df_features = pd.read_csv(filepath + '\\features_sampled_reduced.csv', index_col=0)
        if group == False:
            print('Automated grouping: False')
            df_features = pd.read_csv(filepath + '\\features_sampled.csv', index_col=0)
    
    if remove_RS == True:
        print('Remove RS: True')
        df_features = feature_engineering.remove_RS(df_features)
        
    if sequence == True:
        print('Use sequence features: True')
        df_sequence_micro = pd.read_csv(filepath + '\extra_features\PrNrn_sequence_micro.csv', index_col=0)
        df_sequence_macro = pd.read_csv(filepath + '\extra_features\PrNrn_sequence_macro.csv', index_col=0)
        df_features = df_features.join(df_sequence_micro).join(df_sequence_macro)
        
    print('Dataset consists of ' + str(len(df_features)) + ' with ' + str(len(df_features.columns)) + ' features')
    print()
    
    return df_features
        
