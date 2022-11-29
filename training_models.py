# Training models

## Libraries

import ResIndex
import methods1_ml as ml
import methods2_nn as nn
import methods2_ft as ft
import methods3_tl as tl
import methods_evaluation_metrics as eval
import methods_data_import_and_preprocessing as pr
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
import spacy
spacy.cli.download("en_core_web_sm")
spacy.cli.download("en_core_web_sm")

# For reproducibility
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

"""## Data preparation"""

#import of the preprocessed datasets
abs_path = "/newstorage5/aromagno"
path = abs_path + "/paraphrased_datasets"
dataset_list = ["ace"]#, "copd", "ppi"]#, "alhammad", "ghasemi", "goulao", "guinea", "santos", "shahin", "yang"]
dataset = dict()
for df in dataset_list:
  dataset[df] = pd.read_csv(path + "/paraphrased_" + df + ".csv", index_col=0)
  dataset[df] = pd.DataFrame(dataset[df])

#random seed for reproducibility
SEED = [123, 2839, 516, 2383, 273, 1625, 1324, 2791, 7, 1928] #for cross-validation

#parameters
APPROACH = "NN"
CLEAN_TEXT = True
TEXT_PREPROCESSING = True
PARAPHRASING = True
SORTING = True #False, if you don't have a batch_size
TRAIN_SIZE = 0.5
SAMPLING = 1
K = 95

"""## Training with cross validation"""

#creating dataset to save results
all_epochs_res = pd.DataFrame() #results for all epochs
res = pd.DataFrame() #results for best epochs only

#creating dataset to save predictions
pred = pd.DataFrame() #predictions

for i in dataset:

  #dataset[i] = dataset[i].loc[100,:]

  print("\nDATASET", i, "\n")
  j = 0 # fold number

  dataset[i] = pr.maintain_only_text_label(df=dataset[i],
                                           cleantext=CLEAN_TEXT,
                                           paraphrasing=PARAPHRASING,
                                           approach=APPROACH)

  #cross validation k folds
  for h in range(len(SEED)):

    j += 1
    print("\nSplit number", j, "\n")

    #train-test split
    set1, set2 = train_test_split(dataset[i],
                                  train_size=TRAIN_SIZE,
                                  random_state=random.seed(SEED[h]),
                                  shuffle=True,
                                  stratify=dataset[i]['label'])

    for k in range(2): #swapping train and test (kx2 cross-validation)
        res_index = ResIndex.ResIndex(i, j, k+1)

        if k == 0:
            train = set1.copy()
            test = set2.copy()
            print("First iteration")
        else:
            train = set2.copy()
            test = set1.copy()
            print("\nSecond iteration")

        train, test = pr.perform_options(train=train,
                                         test=test,
                                         paraphrasing=PARAPHRASING,
                                         text_preprocessing=TEXT_PREPROCESSING,
                                         sorting=SORTING,
                                         cleantext=CLEAN_TEXT,
                                         batch_size=nn.BATCH_SIZE)

        #PREPARING DATASET

        if APPROACH == "ML":
            X_train, y_train, X_test, y_test = ml.final_ml_preprocessing(train=train,
                                                                         test=test,
                                                                         sampling=SAMPLING,
                                                                         seed=SEED[0])
        elif APPROACH == "NN":
            train_data, valid_data, test_data = nn.final_nn_preprocessing(train=train,
                                                                          test=test,
                                                                          sampling=SAMPLING,
                                                                          seed=SEED[0])
        elif APPROACH == "FT":
            train, test = ft.final_ft_preprocessing(train=train,
                                                    test=test,
                                                    sampling=SAMPLING,
                                                    seed=SEED[0],
                                                    path=abs_path)

        elif APPROACH == "TL":
            df = tl.final_tl_preprocessing(train=train,
                                           test=test,
                                           sampling=SAMPLING,
                                           sorting=SORTING,
                                           seed=SEED[0])

        ### TRAINING

        if APPROACH == "ML":
            res, pred = ml.ml_training(X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test,
                                       y_test=pd.DataFrame(y_test),
                                       pred=pred,
                                       res=res,
                                       res_index=res_index,
                                       seed=SEED[0])
        elif APPROACH == "NN":
            all_epochs_res, res, pred = nn.nn_training(train_data=train_data,
                                                       valid_data=valid_data,
                                                       test_data=test_data,
                                                       dataset=dataset[i],
                                                       res_index=res_index,
                                                       res=res,
                                                       epochs_res=all_epochs_res,
                                                       pred=pred)
        elif APPROACH == "FT":
            res, pred = ft.ft_training(train=train,
                                       test=test,
                                       res_index=res_index,
                                       res=res,
                                       pred=pred)

        elif APPROACH == "TL":
            all_epochs_res, res, pred = tl.tl_training(df=df,
                                                       epochs_res=all_epochs_res,
                                                       res_index=res_index,
                                                       res=res,
                                                       pred=pred)




pred_df = eval.adjust_pred(pred=pred, approach=APPROACH)
#compute metrics at 95% recall
res = res.reindex(columns=res.columns.tolist() + ["true_negative_rate@95", "precision@95", "wss@95"])
for l in range(len(pred_df)):
    res = eval.compute_rank_based_metrics(pred=pred_df.iloc[l],
                                          res=res,
                                          approach=APPROACH,
                                          K=K)

avg_res = eval.compute_avg_std(res=res, approach=APPROACH)


#saving results in a csv file
path = abs_path + "/results"
with open(path + "/approach_" + str(APPROACH) + "_paraphrased_sorted_ace" + ".csv", 'w', encoding = 'utf-8-sig') as f:
    res.to_csv(f)
with open(path + "/approach_" + str(APPROACH) + "_avg_paraphrased_sorted_ace" + ".csv", 'w', encoding = 'utf-8-sig') as f:
    avg_res.to_csv(f)
if APPROACH in ["NN", "TL"]:
    with open(path + "/approach_" + str(APPROACH) + "_all_epochs_paraphrased_sorted_ace" + ".csv", 'w', encoding='utf-8-sig') as f:
        all_epochs_res.to_csv(f)

