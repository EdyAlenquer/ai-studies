from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')
import copy
import numpy as np
from utils.metrics import rmse
from utils.activation_functions import TanH

def grid_search(
    model, 
    params, 
    X, 
    y, 
    valid_set,
    loss_function=rmse,
    lower_is_better=True, 
    verbose=True
):
    
    start = time.time()

    if not lower_is_better:
        loss_function = np.negative(loss_function)

    best_loss = np.inf 
    params_combination_list = list(product(*list(params.values())))

    for trial, combination in enumerate(params_combination_list):
        actual_params = {param: key for param, key in zip(list(params.keys()), list(combination))}
        
        if verbose:
            print(f'Running trial {trial+1}/{len(params_combination_list)}')
        
        instance = copy.deepcopy(model)
        
        instance.add_hidden_layer(n_neurons=actual_params['n_neurons'], activation_function=TanH())

        instance.fit(
            X,
            y,
            valid_set=valid_set,
            **actual_params
        )

        # Getting metrics
        X_val, y_val = valid_set
        val_preds =  instance.predict(X_val)
        loss = loss_function(y_val, val_preds)

        if verbose:
            print('Metric:', loss, '| Parameters:', actual_params)

        # Storing if best metric
        if loss < best_loss:
            best_loss = loss
            best_params = actual_params
            if verbose:
                print('New best metric!')

        trial+=1
        print('-'*50)

    end = time.time()
    print('-'*50)
    print('Grid Search Completed in ', round(end - start, 4), 'seconds')
    print('Selected parameters:')
    print(best_params)

    return best_params
