import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.utils import to_categorical
from keras.utils import np_utils

import importlib

from helper_functions import SQA_preprocessing as SQA_prepro
from helper_functions import feature_preprocessing as feature_prepro
from helper_functions import methods_sampling as sampl_meth
from helper_functions import keras_NN as keras_NN
from helper_functions import evaluation_metrics as eval_metr
from helper_functions import feature_engineering as feature_eng
from helper_functions import parametergrids

from sklearn.preprocessing import MinMaxScaler
from collections import Counter

importlib.reload(SQA_prepro)
importlib.reload(feature_prepro)
importlib.reload(keras_NN)
importlib.reload(sampl_meth)
importlib.reload(eval_metr)


# Method to prepare a dataset before training a classification algorithm
def prepare_dataset(model_type, df_PrNrn, df_sqa, meta_paramgrid):
    
    # Copy parameters from parametergrid
    cut_off_high = meta_paramgrid['cut_off_high']
    cut_off_low = meta_paramgrid['cut_off_low']
    delete_begin = meta_paramgrid['delete'][0]
    delete_end = meta_paramgrid['delete'][1]
    threshhold_late = meta_paramgrid['threshhold_late']
    target_col = meta_paramgrid['target_col']
    residual_col = target_col + '_cat'
    sampling = meta_paramgrid['sampling']
    sample_frac = meta_paramgrid['sample_frac']
    cv_num = meta_paramgrid['cv_num']
    
    # Optional random sampling of dataset
    if sample_frac < 1:
        print('Dataset is randomly sampled down to ' + str(sample_frac) + ' times size.')
        df_PrNrn = df_PrNrn.sample(frac=sample_frac)
    
    # Perform percentage/frequency cut-off
    df_PrNrn = feature_eng.percentage_cut_off(cut_off_low, cut_off_high, df_PrNrn)
    print('Performed cut-off with low: ' + str(cut_off_low) + ' cut-off and high: ' + str(cut_off_high) + ' cut-off.')
    
    # Save the shape of the input dataframe (# of columns)
    input_shape = len(df_PrNrn.columns)

    # Delete begin & end of dataset because SQAs are not correct at the ends of the dataset
    print('Before cutting, the dataset contains ' + str(len(df_PrNrn)) + ' instances')
    df_PrNrn.sort_index(inplace=True)
    df_PrNrn = df_PrNrn.iloc[delete_begin:-delete_end,:]
    print('After cutting, the dataset contains ' + str(len(df_PrNrn)) + ' instances')

    # Start SQA-preprocessing
    print('Start SQA-Preprocessing with target: ' + str(target_col) + ' and threshhold: ' + str(threshhold_late))
    
    # If regression, than no label binarization
    if (model_type == 'regression'):
        print('Prepare Dataset for Regression')
        df_SQA = df_sqa.loc[:, [target_col]]
    else:  
        print('Prepare Dataset for Classification')
        df_SQA = SQA_prepro.get_binary_SQA_Cats(df_sqa.loc[:, [target_col]], target_col, threshhold_late)
        df_SQA.drop([target_col], axis=1, inplace=True)
    
    # Join with SQA information
    print('Perform join...')
    df_joined = df_PrNrn.join(df_SQA, on='Kennnummer')
    df_joined.dropna(inplace=True)
    print('Nach join mit SQA noch ' + str(len(df_joined)) + '/' + str(len(df_PrNrn)) + ' Fahrzeugen im Dataset')
    
    #Reset index to perform KFold CV
    df_joined.reset_index(inplace=True, drop=True)

    return df_joined, input_shape


# Method to train and predict a neural network classifier
def perform_binary_classification (df_PrNrn, df_sqa, meta_paramgrid, model_paramgrid):
    
    # Copy paramter from parameter grids
    layer_depth = model_paramgrid['layer_depth']
    layer_architecture = model_paramgrid['layer_architecture']
    input_activation_function = model_paramgrid['input_activation_function']
    input_neurons = model_paramgrid['input_neurons']
    dropout = model_paramgrid['dropout']
    hidden_neurons = model_paramgrid['hidden_neurons']
    hidden_activation_function = model_paramgrid['hidden_activation_function']
    output_activation_function = model_paramgrid['output_activation_function']
    output_neurons = model_paramgrid['output_neurons']
    optimizer = model_paramgrid['optimizer']
    loss = model_paramgrid['loss']
    metric = model_paramgrid['metric']
    epochs = model_paramgrid['epochs']
    batch_size = model_paramgrid['batch_size']

    cut_off_high = meta_paramgrid['cut_off_high']
    cut_off_low = meta_paramgrid['cut_off_low']
    threshhold_late = meta_paramgrid['threshhold_late']
    target_col = meta_paramgrid['target_col']
    residual_col = target_col + '_cat'
    sampling = meta_paramgrid['sampling']
    sample_frac = meta_paramgrid['sample_frac']
    cv_num = meta_paramgrid['cv_num']
    pred_certainty = meta_paramgrid['pred_certainty']

    # Call method to prepare the dataset
    df_joined, input_shape = prepare_dataset('binary', df_PrNrn, df_sqa, meta_paramgrid)

    # KFold CV
    n_split=cv_num
    
    # Create empty lists to store the measures
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []
    cms = []
    models = []
    y_preds = []
    y_tests = []
    model_results = []
    
    # Perform Train-Validation-Split
    c = 0
    
    for train_index, test_index in KFold(n_split, shuffle=True).split(df_joined):
        
        print()
        print(' -- Convolution ' + str(c+1) + '/' + str(n_split) + ': ')
        c = c+1
        
        train = df_joined.iloc[train_index]
        test = df_joined.iloc[test_index]

        # Balance the training data
        train_balanced = sampl_meth.sample_data(train, residual_col, sampling)

        # Split features from label
        X_train = train_balanced.iloc[:,:-1].values
        y_train = train_balanced.iloc[:,-1:].values
        X_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1:].values
        
        y_train=y_train.astype('int')
        y_test=y_test.astype('int')
        
        # Train & run model
        print()
        print('Start Model Training')
        
        # Call method to initialize model with the parameters provided
        model = keras_NN.create_model(layer_depth, layer_architecture, input_activation_function, input_shape, input_neurons, dropout, hidden_neurons, hidden_activation_function, 
        output_activation_function, output_neurons)
        
        print()
        print('Finished Model Training')
        print()
        
        # Call method to train the model with the parameters provided
        # Returns predictions on the rest dataset
        model_trained, predictions, results = keras_NN.train_model(model, optimizer, loss, metric, epochs, batch_size, X_train, y_train, X_test, y_test)
    
        # Call method to calculate binary performance metrics
        print('Calculate metrics...')
        f1_score, f2_score, precision_score, recall_score, cm = eval_metr.binary_classification_metrics(predictions, y_test, pred_certainty)
        
        # Store results and measures
        f1_scores.append(f1_score)
        f2_scores.append(f2_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        cms.append(cm)
        models.append(model_trained)
        y_preds.append(predictions)
        y_tests.append(y_test)
        model_results.append(results)
        
    return models, f1_scores, f2_scores, precision_scores, recall_scores, cms, y_preds, y_tests, model_results
    
    
    
# Method to perform a regression prediction
def perform_regression (df_PrNrn, df_sqa, meta_paramgrid, model_paramgrid):
    
    # Copy paramter from parameter grids
    cut_off_high = meta_paramgrid['cut_off_high']
    cut_off_low = meta_paramgrid['cut_off_low']
    delete = meta_paramgrid['delete']
    threshhold_late = meta_paramgrid['threshhold_late']
    target_col = meta_paramgrid['target_col']
    residual_col = target_col + '_cat'
    sampling = meta_paramgrid['sampling']
    sample_frac = meta_paramgrid['sample_frac']
    cv_num = meta_paramgrid['cv_num']
    
    layer = model_paramgrid['layer_depth']
    layer_architecture = model_paramgrid['layer_architecture']
    input_activation_function = model_paramgrid['input_activation_function']
    input_neurons = model_paramgrid['input_neurons']
    dropout = model_paramgrid['dropout']
    hidden_neurons = model_paramgrid['hidden_neurons']
    hidden_activation_function = model_paramgrid['hidden_activation_function']
    output_activation_function = model_paramgrid['output_activation_function']
    output_neurons = model_paramgrid['output_neurons']
    optimizer = model_paramgrid['optimizer']
    loss = model_paramgrid['loss']
    metric = model_paramgrid['metric']
    epochs = model_paramgrid['epochs']
    batch_size = model_paramgrid['batch_size']
    
    # Call method to prepare dataset
    df_joined, input_shape = prepare_dataset('regression', df_PrNrn, df_sqa, meta_paramgrid)

    # KFold CV
    n_split=cv_num
    
    # Create empty lists to store measures
    mses = []
    rmses = []
    maes = []
    mapes = []
    raes = []
    r_squareds = []
    adj_r_squareds = []
    median_abs_errors = []
    models = []
    y_preds = []
    y_tests = []
    model_results = []
    
    c = 0
    
    # K-Fold CV
    for train_index, test_index in KFold(n_split, shuffle=True).split(df_joined.iloc[:,:-1]):
        
        print()
        print(' -- Convolution ' + str(c+1) + '/' + str(n_split) + ': ')
        c = c+1
        print()
        
        # Train-test-Split
        train = df_joined.iloc[train_index]
        test = df_joined.iloc[test_index]
               
        # Split features from label
        X_train = train.iloc[:,:-1].values
        y_train = train.iloc[:,-1:].values
        X_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1:].values
        
        y_train=y_train.astype('int')
        y_test=y_test.astype('int')
        
        y = df_joined.iloc[:,-1:]
        
        # Scale label with MinMaxScaler()
        scaler_y = MinMaxScaler()
        print(scaler_y.fit(y))
        y_train_scaled = scaler_y.transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        # Train & Run Model
        print('Start Model Training')
        
        # Call method to initiate model with the provided parameters
        model = keras_NN.create_model(layer, layer_architecture, input_activation_function, input_shape, input_neurons, dropout, hidden_neurons, 
        hidden_activation_function, output_activation_function, output_neurons)
        
        print('y_train scaled mean: ' + str(y_train_scaled.mean()))
        
        # Call method to predict on model with the provided parameters
        model_trained, predictions, results = keras_NN.train_model(model, optimizer, loss, metric, epochs, batch_size, X_train, y_train, X_test, y_test)
    
        # Calculate binary metrics
        print('Calculate metrics...')
        
        print('predictions mean: ' + str(predictions.mean()))
        
        print('y_pred inverse_scaled mean: ' + str(predictions.mean()))
        
        # Store results
        y_preds.append(predictions)
        y_tests.append(y_test)
        
        # Call method to calculate regression performance measures
        mse, rmse, mae, mape, rae, r_squared, adj_r_squared, median_abs_error = eval_metr.regression_metrics(predictions, y_test, X_test)
    
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)
        raes.append(rae)
        r_squareds.append(r_squared)
        adj_r_squareds.append(adj_r_squared)
        median_abs_errors.append(median_abs_error)
        models.append(model_trained)
        model_results.append(results)
        
    return models, mses, rmses, maes, mapes, raes, r_squareds, adj_r_squareds, median_abs_errors, y_preds, y_tests, model_results
    

# Method to perform a classification prediction using a random forest classifier    
def perform_binary_classification_RF (df_PrNrn, df_sqa, meta_paramgrid, model_paramgrid):
    
    # Copy parameters from parameter grids
    cut_off_high = meta_paramgrid['cut_off_high']
    cut_off_low = meta_paramgrid['cut_off_low']
    threshhold_late = meta_paramgrid['threshhold_late']
    target_col = meta_paramgrid['target_col']
    residual_col = target_col + '_cat'
    sampling = meta_paramgrid['sampling']
    sample_frac = meta_paramgrid['sample_frac']
    cv_num = meta_paramgrid['cv_num']
    pred_certainty = meta_paramgrid['pred_certainty']

    estimators = model_paramgrid['n_estimators']
    crit = model_paramgrid['criterion']
    min_sampl_split = model_paramgrid['min_samples_split']
    
    # Call method to prepare dataset
    df_joined, input_shape = prepare_dataset('binary', df_PrNrn, df_sqa, meta_paramgrid)
    
    n_split=cv_num
    
    # Create empty lists to store results
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []
    cms = []
    models = []
    y_preds = []
    y_tests = []
    model_results = []
    
    c = 0
    
    # KFold CV
    for train_index, test_index in KFold(n_split, shuffle=True).split(df_joined.iloc[:,:-1]):

        print()
        print(' -- Convolution ' + str(c+1) + '/' + str(n_split) + ': ')
        c = c+1
        print()
        
        # Train-Test-Split
        train = df_joined.iloc[train_index]
        test = df_joined.iloc[test_index]

        # Balance the training data
        train_balanced = sampl_meth.sample_data(train, residual_col, sampling)

        # Split features from label
        X_train = train_balanced.iloc[:,:-1].values
        y_train = train_balanced.iloc[:,-1:].values
        X_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1:].values
        
        y_train=y_train.astype('int')
        y_test=y_test.astype('int')
        
        # Train & Run Model
        print('Start Model Training')
        
        # Initialize model
        clf = RandomForestClassifier(n_estimators = estimators, criterion=crit, min_samples_split=min_sampl_split)        
        clf.fit(X_train, y_train.ravel())
        
        # Make predictions with the model
        predictions = clf.predict(X_test)
        
        # Calculate binary metrics
        print('Calculate metrics...')
        f1_score, f2_score, precision_score, recall_score, cm = eval_metr.binary_classification_metrics(predictions, y_test, pred_certainty)
    
        # Store performance measures/results
        f1_scores.append(f1_score)
        f2_scores.append(f2_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        cms.append(cm)
        y_preds.append(predictions)
        y_tests.append(y_test)

        
    return models, f1_scores, f2_scores, precision_scores, recall_scores, cms, y_preds, y_tests, model_results
    

# Method to perform a classification with an XGBoost classifier   
def perform_binary_classification_XGBoost (df_PrNrn, df_sqa, meta_paramgrid, model_paramgrid):
    
    # Copy parameters from parameter grid
    cut_off_high = meta_paramgrid['cut_off_high']
    cut_off_low = meta_paramgrid['cut_off_low']
    threshhold_late = meta_paramgrid['threshhold_late']
    target_col = meta_paramgrid['target_col']
    residual_col = target_col + '_cat'
    sampling = meta_paramgrid['sampling']
    sample_frac = meta_paramgrid['sample_frac']
    cv_num = meta_paramgrid['cv_num']
    pred_certainty = meta_paramgrid['pred_certainty']
    
    booster = model_paramgrid['booster']
    eta = model_paramgrid['eta']
    gamma = model_paramgrid['gamma']
    max_depth = model_paramgrid['max_depth']
    lambda_para = model_paramgrid['lambda']
    alpha_para = model_paramgrid['alpha']
    tree_method = model_paramgrid['tree_method']
    num_parallel_tree = model_paramgrid['num_parallel_tree']

    # Prepare dataset
    df_joined, input_shape = prepare_dataset('binary', df_PrNrn, df_sqa, meta_paramgrid)
    
    # KFold CV
    n_split=cv_num
    
    # Create empty lists to store results
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []
    cms = []
    models = []
    y_preds = []
    y_tests = []
    model_results = []
    
    c = 0    
    
    # K-Fold CV
    for train_index, test_index in KFold(n_split, shuffle=True).split(df_joined.iloc[:,:-1]):

        print()
        print(' -- Convolution ' + str(c+1) + '/' + str(n_split) + ': ')
        c = c+1
        print()
        
        # Train-test Split
        train = df_joined.iloc[train_index]
        test = df_joined.iloc[test_index]

        # Balance the training data
        train_balanced = sampl_meth.sample_data(train, residual_col, sampling)

        # Split features from label
        X_train = train_balanced.iloc[:,:-1].values
        y_train = train_balanced.iloc[:,-1:].values
        X_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1:].values
        
        y_train=y_train.astype('int')
        y_test=y_test.astype('int')
        
        # Train & Run Model
        print('Start Model Training')
        
        # Initialize model
        clf = XGBClassifier(nthread=-1, booster=booster, eta=eta, gamma=gamma, max_depth=max_depth, reg_lambda=lambda_para, reg_alpha=alpha_para, tree_method=tree_method, num_parallel_tree=num_parallel_tree)        
        clf.fit(X_train, y_train.ravel())
        
        # Make predictions with the model on test data
        predictions = clf.predict(X_test)
        
        # Calculate binary metrics
        print('Calculate metrics...')
        f1_score, f2_score, precision_score, recall_score, cm = eval_metr.binary_classification_metrics(predictions, y_test, pred_certainty)
        
        # Store performance measures/results
        f1_scores.append(f1_score)
        f2_scores.append(f2_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        cms.append(cm)
        y_preds.append(predictions)
        y_tests.append(y_test)
        
    return models, f1_scores, f2_scores, precision_scores, recall_scores, cms, y_preds, y_tests, model_results    
    

# Method to provide an initialized neural network classifier 
def get_binary_NN_clf(input_shape):
    
    # Load the parametergrid
    model_paramgrid = parametergrids.model_paramgrid = parametergrids.paramgrid_binary_NN[0]
    
    # Copy the params from the parametergrid
    layer_depth = model_paramgrid['layer_depth']
    layer_architecture = model_paramgrid['layer_architecture']
    input_activation_function = model_paramgrid['input_activation_function']
    input_neurons = model_paramgrid['input_neurons']
    dropout = model_paramgrid['dropout']
    hidden_neurons = model_paramgrid['hidden_neurons']
    hidden_activation_function = model_paramgrid['hidden_activation_function']
    output_activation_function = model_paramgrid['output_activation_function']
    output_neurons = model_paramgrid['output_neurons']
    optimizer = model_paramgrid['optimizer']
    loss = model_paramgrid['loss']
    metric = model_paramgrid['metric']
    epochs = model_paramgrid['epochs']
    batch_size = model_paramgrid['batch_size']
    
    # Initialize the model with the parameters
    model = keras_NN.create_model(layer_depth, layer_architecture, input_activation_function, input_shape, input_neurons, dropout, hidden_neurons, hidden_activation_function, 
        output_activation_function, output_neurons)
    
    # Compile the model
    model.compile(optimizer = optimizer, loss = loss, metrics = [metric])
      
    return model
     
     
    
    
    
    
    
    
    