# -*- coding: utf-8 -*-
"""Training_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PPsFOvrUiWinbszvytz9Uh5C6iktSS0g

# Training models

## Libraries
"""

import os
import ResIndex
import methods1_ml as ml
import methods2_nn as nn
import methods3_tl as tl
import methods_evaluation_metrics as eval
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import spacy
spacy.cli.download("en_core_web_sm")

"""## Data preparation"""

#import of the preprocessed datasets
path = os.getcwd() + "\datasets\preprocessed_datasets"
dataset_list = ["alhammad"] #"ace", "copd", "ppi", "alhammad", "ghasemi", "goulao", "guinea", "santos", "shahin", "yang"]
dataset = dict()
for df in dataset_list:
  dataset[df] = pd.read_csv(path + "/preprocessed_" + df + ".csv", index_col=0)

#random seed for reproducibility
SEED = [1009]#, 2839, 516, 2383, 273, 1625, 1324, 2791, 7, 1928] #for cross-validation

#parameters
APPROACH = 1
CLEAN_TEXT = True
TRAIN_SIZE = 0.5
SAMPLING = 1
K = 95

"""## Training with cross validation"""

#creating dataset to save results
res = pd.DataFrame()

#creating datasets and dictionaries to save predictions
pred = pd.DataFrame() #ML predictions
all_epochs_res = pd.DataFrame() #NN predictions
best_train_preds_dict = {} #NN predictions
best_valid_preds_dict = {} #NN predictions
test_preds_dict = {} #NN predictions



for i in dataset:

  dataset[i] = dataset[i].loc[:40,:]

  print("\nDATASET", i, "\n")
  j = 0 # fold number

  if APPROACH in [2, 3]:
      dataset[i] = nn.maintain_only_text_label(df = dataset[i],
                                               clean_text = CLEAN_TEXT)

  #cross validation k folds
  for h in range(len(SEED)):

    j += 1
    print("\nSplit number", j, "\n")

    #train-test split
    set1, set2 = train_test_split(dataset[i],
                                  train_size=TRAIN_SIZE,
                                  random_state=random.seed(SEED[0]),
                                  shuffle=True,
                                  stratify=dataset[i]['label'])

    for k in range(2): #swapping train and test (kx2 cross-validation)
        res_index = ResIndex.ResIndex(i, j, k+1)

        if k == 0:
            train = set1
            test = set2
            print("First iteration")
        else:
            train = set2
            test = set1
            print("\nSecond iteration")

        #PREPARING DATASET

        if APPROACH == 1:
            X_train, y_train, X_test, y_test = ml.final_ml_preprocessing(train = train,
                                                                         test = test,
                                                                         sampling = SAMPLING,
                                                                         seed = SEED[0])
        elif APPROACH == 2:
            train_data, valid_data, test_data = nn.final_nn_preprocessing(train = train,
                                                                          test = test,
                                                                          sampling = SAMPLING,
                                                                          seed = SEED[0])
        elif APPROACH == 3:
            df = tl.final_tl_preprocessing(train = train,
                                           test = test,
                                           sampling = SAMPLING,
                                           seed = SEED[0])

        ### TRAINING

        if APPROACH == 1:
            res, pred = ml.ml_training(X_train = X_train,
                                       y_train = y_train,
                                       X_test = X_test,
                                       y_test = pd.DataFrame(y_test),
                                       pred = pred,
                                       res = res,
                                       res_index = res_index,
                                       seed = SEED[0])
        elif APPROACH == 2:
            all_epochs_res, res, pred = nn.nn_training(train_data = train_data,
                                                       valid_data = valid_data,
                                                       test_data = test_data,
                                                       dataset=dataset[i],
                                                       res_index = res_index,
                                                       res = res,
                                                       epochs_res = all_epochs_res,
                                                       pred = pred)

        elif APPROACH == 3:
            all_epochs_res, res, pred = tl.tl_training(df=df,
                                                       epochs_res = all_epochs_res,
                                                       res = res,
                                                       pred=pred,
                                                       res_index = res_index)




pred_df = eval.adjust_pred(pred = pred, approach = APPROACH)
#compute metrics at 95% recall
res = res.reindex(columns=res.columns.tolist() + ["true_negative_rate@95", "precision@95", "wss@95"])
for l in range(len(pred_df)):
    res = eval.compute_rank_based_metrics(pred = pred_df.iloc[l],
                                          res = res,
                                          approach = APPROACH,
                                          K = K)

avg_res = eval.compute_avg_std(res = res, approach = APPROACH)


#saving results in a csv file
path = os.getcwd() + "\\results"
with open(path + "\\approach_" + str(APPROACH) + "_results.csv", 'w', encoding = 'utf-8-sig') as f:
  res.to_csv(f)
with open(path + "\\approach_" + str(APPROACH) + "_avg_results.csv", 'w', encoding = 'utf-8-sig') as f:
  avg_res.to_csv(f)
if APPROACH in [2, 3]:
    with open(path + "\\approach_" + str(APPROACH) + "_all_epochs_results.csv", 'w', encoding='utf-8-sig') as f:
        all_epochs_res.to_csv(f)

