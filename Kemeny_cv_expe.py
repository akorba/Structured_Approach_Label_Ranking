#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import time

from sklearn.model_selection import RepeatedKFold , GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

from misc_functions import ranking_format_sorted
from kemeny_hamming_embeddings import KemenyEmbed_no_offset_unisign,KemenyInvert_unisign,HammingEmbed

from sklearn.metrics import hamming_loss
from sklearn.pipeline import Pipeline
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from itertools import combinations

import operalib as ovk #for gradient based ridge learning when output embedding is large



#### parameters for running code ####
regressor = 'onorma' # can use 'kernel_ridge', 'rf', 'onorma' or 'knn'
datasets_choice = 'supplementary' #  can use 'supplementary' and 'additionals'
base_data_path = 'data/'

random_state = 1234
# Choose a dataset

if datasets_choice == 'main_paper':
    dataset_grid =  ['iris','vehicle','glass','authorship','vowel','wine']  # main page paper
elif datasets_choice == 'supplementary':
    dataset_grid = ['bodyfat','calhousing','cpu-small','pendigits','segment','wisconsin','fried']
elif datasets_choice == 'sushi':
    dataset_grid = ['sushi_one_hot']
else:
    print('unknown dataset choice')
    exit()




# hyperparameters for grid search cv:


if regressor == 'kernel_ridge':
    alpha_grid,gamma_grid = [],[]
    for i in range(6):
        alpha_grid += [10.**-i,5.*(10.**-i)]
        gamma_grid += [10. ** -i, 5. * (10. ** -i)]
    parameters = {'clf__alpha' : alpha_grid, 'clf__gamma' : gamma_grid, 'clf__kernel': ('rbf',)}
    pipeline = Pipeline( [('clf',KernelRidge())])
elif regressor == 'knn':
    alpha_grid = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]
    parameters = {'clf__n_neighbors': alpha_grid}
    pipeline = Pipeline([('clf', KNeighborsRegressor())])
elif regressor =='rf':
    parameters = {}
    pipeline = Pipeline([('clf', RandomForestRegressor(n_estimators = 50,max_depth = 50,n_jobs = -1))])
elif regressor == 'onorma':
    gamma_grid = [.0000002,.000002,.00002]
    learning_rate_grid = [ovk.InvScaling(3.),ovk.InvScaling(.5)]
    lbda_grid = [.01]
    parameters = {'clf__gamma': gamma_grid,'clf__learning_rate': learning_rate_grid,'clf__lbda':lbda_grid}
    pipeline = Pipeline([('clf', ovk.ONORMA())])

n_iter = 0
dico_all_results = {}
# any loop could be parallelized
t_1 = time.time()
for dataset_choice in dataset_grid:

    ######################## Loading dataset ########################
    dataset_path = base_data_path + dataset_choice + '.txt'

    dataset = pd.read_csv(dataset_path)
    n = len(dataset)
    features = dataset.drop('ranking', axis=1)
    dataset['label'] = dataset.ranking.map(ranking_format_sorted)

    ##################### Compute Kemeny embeddings ################
    dataset['kemeny'] = dataset.label.map(KemenyEmbed_no_offset_unisign)
    kemeny_labels = dataset['kemeny'].apply(pd.Series)
    kemeny_labels = kemeny_labels.rename(columns=lambda x: 'position_' + str(x))

    ################## Build model in the ILP solver : #############
    # This step is dataset dependent (depends on the number of items)

    len_Kemeny_embedding = kemeny_labels.shape[1]
    n_items = len(dataset['label'].loc[0])
    dico_corresp_sigma = {}
    dico_corresp_index_sigma = {}
    L_best_parameters = []
    for idx, (i, j) in enumerate(combinations(range(n_items), 2)):
        dico_corresp_sigma[(i, j)] = 's'  # sigma
        dico_corresp_sigma[(j, i)] = 'i'  # sigma inv
        dico_corresp_index_sigma[(i, j)] = idx
        dico_corresp_index_sigma[(j, i)] = idx

    simplex = CyClpSimplex()

    sigma = simplex.addVariable('sigma', len_Kemeny_embedding, isInt=True)
    sigma_inv = simplex.addVariable('sigma_inv', len_Kemeny_embedding, isInt=True)

    for (i, j, k) in combinations(range(n_items), 3):

        idx_sig1 = dico_corresp_index_sigma[(i, j)]
        idx_sig2 = dico_corresp_index_sigma[(j, k)]
        idx_sig3 = dico_corresp_index_sigma[(k, i)]
        if dico_corresp_sigma[(i, j)] == 's':
            sig_1 = sigma[idx_sig1]
        else:
            sig_1 = sigma_inv[idx_sig1]
        if dico_corresp_sigma[(j, k)] == 's':
            sig_2 = sigma[idx_sig2]
        else:
            sig_2 = sigma_inv[idx_sig2]
        if dico_corresp_sigma[(k, i)] == 's':
            sig_3 = sigma[idx_sig3]
        else:
            sig_3 = sigma_inv[idx_sig3]

        simplex += sig_1 + sig_2 + sig_3 >= 1

    simplex += sigma + sigma_inv == 1
    simplex += sigma <= 1
    simplex += sigma >= 0

    ############ scores with Repeated KFold 5 times as in Cheng ###
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)

    # outer cv evaluation loop
    L_results_kendall_distance = []
    L_results_kendall_coeff = []
    L_hamming_loss = []
    for idx_train,idx_test in rkf.split(range(n)):
        X_train, X_test= features.loc[idx_train],features.loc[idx_test]
        y_train, y_test = kemeny_labels.loc[idx_train] , kemeny_labels.loc[idx_test]

        # inner cv hyper parameter optimization
        grid_search = GridSearchCV(pipeline, parameters, cv = 5, n_jobs=-1)
        grid_search.fit(X_train,y_train)

        best_params = grid_search.best_params_
        regr = grid_search.best_estimator_
        L_best_parameters += [best_params]


        ##################### feature space prediction (in F_Y) ####################


        pred_y_test = regr.predict(X_test)

        ######### pre-image computation on the test set (embedding dependent) ######
        L_pred = []
        L_pred_list = []
        # for loop could be set in parallel
        for idx_test in range(len(y_test)):

            cost_vector = CyLPArray(-pred_y_test[idx_test, :])  # example

            simplex.objective = 2 * (cost_vector + 0.5) * sigma

            # solve using branch and bound :
            cbcModel = simplex.getCbcModel()
            verbose = cbcModel.branchAndBound()

            solution = cbcModel.primalVariableSolution['sigma']
            L_pred += [solution]




        ## Store results


        predictions = pd.Series(L_pred).map(KemenyInvert_unisign)
        predictions_list = predictions.apply(lambda x:x.tolist())

        real_rankings = dataset['label'].loc[y_test.index] #pd.Series(y_test).map(KemenyInvert_unisign)


        out_emb_pred = predictions_list.map(HammingEmbed)
        out_emb_pred = np.asarray([i.ravel() for i in out_emb_pred])
        out_emb_real = real_rankings.map(HammingEmbed)
        out_emb_real = np.asarray([i.ravel() for i in out_emb_real])

        local_Hamming_loss = hamming_loss(out_emb_real, out_emb_pred)
        L_hamming_loss += [local_Hamming_loss]

        L_kendall_tau_coeff = [kendalltau(pred,real).correlation for ((_,pred),(_,real)) in
                               zip(predictions.iteritems(),real_rankings.iteritems())]
        mean_kendall_tau_coeff = np.mean(L_kendall_tau_coeff)
        result_kendall_tau = np.mean(np.sum(np.abs(y_test.as_matrix() - np.asarray(L_pred)),axis = 1))
        result_kendall_tau_normalized = result_kendall_tau/y_test.shape[1]


        L_results_kendall_distance += [result_kendall_tau_normalized]
        L_results_kendall_coeff += [mean_kendall_tau_coeff]


        n_iter += 1





    print('kendall correlation results : '+str(dataset_choice))
    print('mean result :'+str(np.mean(L_results_kendall_coeff)))
    print('std resul :'+str(np.std(L_results_kendall_coeff)))
    print('hamming loss : ' + str(np.mean(L_hamming_loss)))

    local_results = {'best_parameters':L_best_parameters,
                     'mean_result_kendalltau':np.mean(L_results_kendall_coeff),
                     'std_kendalltau' : np.std(L_results_kendall_coeff),
                     'mean hamming loss': np.mean(L_hamming_loss),
                     'std hamming loss': np.std(L_hamming_loss),
                     'estimator':regressor}
    dico_all_results[dataset_choice] = local_results


t_end = time.time()

print('time :'+str(t_end - t_1))

name_saved_file = regressor+ '_Kemeny_'+ datasets_choice+ '.pkl'


try:
    pickle.dump(dico_all_results,open('saved_results/'+name_saved_file,'wb'))
except:
    pickle.dump(dico_all_results, open( name_saved_file, 'wb'))
    print('no existing folder saved_results, results were saved in the current folder instead')
