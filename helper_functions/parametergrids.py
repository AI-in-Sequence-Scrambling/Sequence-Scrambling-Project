# This is a set of final parameters for both meta parameters, e.g. sampling algorithms as well as hyperparameters for specific algorithms

from sklearn.model_selection import ParameterGrid

params_binary_NN = {
              'layer_depth': [8],
              'layer_architecture': [['Dense', 'Dropout']],
              'input_activation_function': ['relu'], 
              'input_neurons': [50], 
              'dropout': [0.5],
              'hidden_neurons': [50], 
              'hidden_activation_function': ['relu'], 
              'output_activation_function': ['sigmoid'], 
              'output_neurons': [1], 
              'optimizer': ['Adam'], 
              'loss': ['binary_crossentropy'], 
              'metric': ['accuracy'],
              'epochs': [4], 
              'batch_size': [50] 
              }

paramgrid_binary_NN = ParameterGrid(params_binary_NN)


param_grid_binary_RF = {
              'n_estimators': [500],
              'criterion': ['entropy'],
              'min_samples_split': [10]
              }

paramgrid_binary_RF = ParameterGrid(param_grid_binary_RF)


param_grid_binary_XGB = {
              'booster': ['gbtree'],
              'eta': [0.3],
              'gamma': [0.1],
              'max_depth': [6],
              'lambda': [1],
              'alpha': [0],
              'tree_method': ['auto'],
              'num_parallel_tree': [3]
              }

paramgrid_binary_XGB = ParameterGrid(param_grid_binary_XGB)


meta_params = {
              'models': [''],
              'cut_off_high': [0.995],
              'cut_off_low': [0.005],
              'delete': [[2000,2000]],
              'threshhold_late': [400], 
              'target_col': ['delta_A600_E100'],  
              'sampling': ['SMOTEOver'], 
              'sample_frac': [1],
              'cv_num': [3],
              'pred_certainty': [0.5]
                         }

meta_paramgrid = ParameterGrid(meta_params)


param_grid_regression_NN = {
              'layer_depth': [5],
              'layer_architecture': [['Dense', 'Dropout']],
              'input_activation_function': ['relu'], 
              'input_neurons': [500], 
              'dropout': [0.3],
              'hidden_neurons': [500], 
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