# This is a set of parameter grids to perform parameter grid searches on both meta parameters, e.g. sampling algorithms as well as hyperparameter searches for specific algorithms

from sklearn.model_selection import ParameterGrid

params_binary_NN = {
              'layer_depth': [4,8],
              'layer_architecture': [['Dense', 'Dropout']],
              'input_activation_function': ['relu'], 
              'input_neurons': [50, 250, 500], 
              'dropout': [0.3, 0.5],
              'hidden_neurons': [50, 250, 500], 
              'hidden_activation_function': ['relu', 'sigmoid'], 
              'output_activation_function': ['sigmoid'], 
              'output_neurons': [1], 
              'optimizer': ['Adam'], 
              'loss': ['binary_crossentropy'], 
              'metric': ['accuracy'],
              'epochs': [4,8], 
              'batch_size': [50, 250, 500] 
              }

paramgrid_binary_NN = ParameterGrid(params_binary_NN)


param_grid_binary_RF = {
              'n_estimators': [1000],
              'criterion': ['entropy'],
              'min_samples_split': [10]
              }

paramgrid_binary_RF = ParameterGrid(param_grid_binary_RF)


param_grid_binary_XGB = {
              'booster': ['dart'],
              'eta': [0.2],
              'gamma': [0.1],
              'max_depth': [6],
              'lambda': [2],
              'alpha': [1],
              'tree_method': ['auto'],
              'num_parallel_tree': [6]
              }

paramgrid_binary_XGB = ParameterGrid(param_grid_binary_XGB)


param_grid_regression_NN = {
              'layer_depth': [5],
              'layer_architecture': [['Dense', 'Dropout']],
              'input_activation_function': ['relu'], 
              'input_neurons': [50], 
              'dropout': [0.3],
              'hidden_neurons': [50], 
              'hidden_activation_function': ['relu'], 
              'output_activation_function': ['linear'], 
              'output_neurons': [1], 
              'optimizer': ['Adam'], 
              'loss': ['mean_squared_error'], 
              'metric': ['accuracy'],
              'epochs': [5], 
              'batch_size': [100]
              }

paramgrid_regression_NN = ParameterGrid(param_grid_regression_NN)


meta_params = {
              'models': [''],
              'cut_off_high': [0.995],
              'cut_off_low': [0.005],
              'delete': [[2000,2000]],
              'threshhold_late': [200], 
              'target_col': ['delta_A600_E100'],  
              'sampling': ['SMOTEENN'], 
              'sample_frac': [1],
              'cv_num': [3],
              'pred_certainty': [0.5]
                         }

meta_paramgrid = ParameterGrid(meta_params)


featureset_params = {
              'remove_RS': [False, True],
              'remove_audit': [False, True],
              'remove_mess': [False, True],
              'cut': [False, 600],
              'cut_area': [""], 
              'group': [True],  
              'base_features': ['all', 'kaco_laco', 'manual'], 
              'sequence': [False, True]
                         }

featureset_paramgrid = ParameterGrid(featureset_params)