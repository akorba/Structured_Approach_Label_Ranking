#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import time

from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import kendalltau

from misc_functions import ranking_format_sorted
from kemeny_hamming_embeddings import HammingEmbed, encode_lehmer, decode_lehmer

from sklearn.metrics import hamming_loss
from sklearn.pipeline import Pipeline



#### parameters for running code ####
regressor = 'knn' # can use 'kernel_ridge'
datasets_choice = 'sushi' #  can use 'supplementary' and 'additionals'

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

    ##################### Compute Hammings embeddings ################
    dataset['lehmer'] = dataset.label.map(encode_lehmer)
    lehmer_labels = dataset['lehmer'].apply(pd.Series)
    lehmer_labels = lehmer_labels.rename(columns=lambda x: 'position_' + str(x))


    ############ scores with Repeated KFold 5 times as in Cheng ###
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)

    # outer cv evaluation loop
    L_results_kendall_distance = []
    L_results_kendall_coeff = []
    L_hamming_loss = []
    L_best_parameters = []
    for idx_train,idx_test in rkf.split(range(n)):
        X_train, X_test= features.loc[idx_train],features.loc[idx_test]
        y_train, y_test = lehmer_labels.loc[idx_train] , lehmer_labels.loc[idx_test]

        # inner cv hyper parameter optimization
        grid_search = GridSearchCV(pipeline, parameters, cv = 5, n_jobs=-1)
        grid_search.fit(X_train,y_train)

        best_params = grid_search.best_params_
        regr = grid_search.best_estimator_
        L_best_parameters += [best_params]


        ##################### feature space prediction (in F_Y) ####################


        pred_y_test = regr.predict(X_test)

        ######### pre-image computation on the test set (embedding dependent) ######



        n2 = pred_y_test.shape[1]


        vect_max_lehmer = np.tile(np.arange(n2), (len(pred_y_test), 1))

        # [[0,1,2,3,..,n],[0,1,2,3,..,n], ... , [0,1,2,3,..,n]]

        proj_0 = np.maximum(pred_y_test, 0)
        proj_pos = np.minimum(proj_0 - vect_max_lehmer, 0)
        projection_pred_y_test = proj_pos + vect_max_lehmer
        round_projection_pred_y_test = np.round(projection_pred_y_test).astype(int)

        pred_y_test_tuple = pd.DataFrame(round_projection_pred_y_test).T.apply(lambda x: tuple(x))
        real_y_test_tuple = y_test.T.apply(lambda x: tuple(x))

        ## Store results

        predictions = pred_y_test_tuple.apply(decode_lehmer)
        predictions_list = predictions.apply(lambda x: list(x))

        real_rankings = real_y_test_tuple.apply(decode_lehmer)


        out_emb_pred = predictions_list.map(HammingEmbed)
        out_emb_pred = np.asarray([i.ravel() for i in out_emb_pred])
        out_emb_real = real_rankings.map(HammingEmbed)
        out_emb_real = np.asarray([i.ravel() for i in out_emb_real])


        local_Hamming_loss = hamming_loss(out_emb_real, out_emb_pred)
        L_hamming_loss += [local_Hamming_loss]

        L_kendall_tau_coeff = [kendalltau(pred,real).correlation for ((_,pred),(_,real)) in
                               zip(predictions.iteritems(),real_rankings.iteritems())]
        mean_kendall_tau_coeff = np.mean(L_kendall_tau_coeff)
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

name_saved_file = regressor+ '_Lehmer_'+ datasets_choice+ '.pkl'


try:
    pickle.dump(dico_all_results,open('saved_results/'+name_saved_file,'wb'))
except:
    pickle.dump(dico_all_results, open(name_saved_file, 'wb'))
    print('no existing folder saved_results, results were saved in the current folder instead')
