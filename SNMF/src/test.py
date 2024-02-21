#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:38 2019

@author: mishugeb
"""
import os
from SNMF import sigpro as sig
import numpy as np
def main():
    min_sig = 5
    max_sig = 5
    reps = 10
    lr = 5e-3

    # l_c = [0., 0.001, 0.01, 0.05,  0.1, 0.25, 0.5, 1.0]
    # l_p = [0., 0.00001, 0.0001, 0.001, 0.01]
    # folds = [1,2,3]

    l_c = [0.0, 0.1]
    l_p = [0.0001]
    folds = ['_all']

    cwd = os.path.dirname(os.path.dirname(os.getcwd()))

    for fold in folds:
        N_train = 100
        # train_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/bootstrapped_sameSplit_sorted/N_{}".format(N_train)
        train_path = os.path.join(cwd, "data/processed/bootstrapped_sameSplit_sorted/N_{}".format(N_train))
        train_data = train_path + '/X_train{}.text'.format(fold)
        train_label = train_path + '/Y_train{}.text'.format(fold)  # one-hot encoded

        N_test = 1000
        test_path = os.path.join(cwd, "data/processed/bootstrapped_sameSplit_sorted/N_{}".format(N_test))
        test_data = test_path + '/X_test{}.text'.format(fold)
        test_label = test_path + '/Y_test{}.text'.format(fold)  # one-hot encoded

        seed_path = "CV/Seeds_3.txt"

        # for k in [3,4,6,7,8]:
        for k in range(min_sig,max_sig+1):
            # Acc_train ; F1_train ; Rec_train ; Lce ; Ltot ; Epochs ; Stability_avg ; Stability_min ; Acc_refit ; F1_refit ; Lrec_refit ; Lce_refit ; Ltot_refit ; Epoch_refit ; Acc_test ; F1_test ; Rec_test
            results = np.zeros((len(l_c), len(l_p), 31))

            for c_idx, lambda_c in enumerate(l_c):
                for p_idx, lambda_p in enumerate(l_p):
                    output_path = 'CV/test_temp/K{}_c{}_p{}_reps{}_f{}'.format(k, c_idx, p_idx, reps, fold)

                    # TRAINING
                    results[c_idx, p_idx, :25] = sig.sigProfilerExtractor("text", output_path, train_data, train_label, minimum_signatures=k, maximum_signatures=k, seeds = seed_path, nmf_replicates=reps, lambda_c= lambda_c, lr= lr, lambda_p= lambda_p, make_decomposition_plots=False)

                    # TEST / VALIDATION
                    results[c_idx, p_idx, 25:28]= sig.test_sigProfilerExtractor("text", output_path, test_data, test_label, minimum_signatures=k, maximum_signatures=k, nmf_replicates=reps, lambda_c= float(lambda_c), lr= float(lr), lambda_p= float(lambda_p), filter = False)

                    f = 'CV/test_temp/SNMF_k{}_f{}.txt'.format(k,fold)


                    with open(f, 'a') as outfile:
                        outfile.write('\n # Lc = {} \t L2 = {} \n'.format(l_c[c_idx], l_p[p_idx]))
                        np.savetxt(outfile, results[c_idx, p_idx,:], fmt='%-8.5f', newline= ' ')
                        outfile.close()


if __name__ == '__main__':
    main()
