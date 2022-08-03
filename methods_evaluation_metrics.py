"""# Evaluation metrics

## Set-based metrics
"""

import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, confusion_matrix, classification_report

def tl_approach_metrics(pred):
    """
    :param pred: EvalPrediction object, which is a named tuple with `predictions` and `label_ids` attributes
    :return:
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return evalmetrics(y_test=labels, pred_test=preds, res=res, labels=[0,1], new_row=row)


def evalmetrics(y_test, pred_test, res, labels, new_row, update_res=False):

    """
    Given a dataset "res" to store the metrics in, a dataset "y_test" with the
    target labels, a dataset "pred_test" with the predicted labels, a list "labels"
    of the unique values of the labels and a dictionary "new_row" with all the
    information to update "res", it returns the dataset "res" updated.
    """


    # accuracy
    accuracy = accuracy_score(y_test, pred_test)
    # precision
    precision = precision_score(y_test, pred_test, average='binary', zero_division=0)
    # recall
    recall = recall_score(y_test, pred_test, average='binary', zero_division=0)
    # f-2 score
    f2 = fbeta_score(y_test, pred_test, average='binary', beta=2, zero_division=0)
    # f-3 score
    f3 = fbeta_score(y_test, pred_test, average='binary', beta=3, zero_division=0)

    # confusion matrix
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    # specificity or true negative rate
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    TNR = TN / (TN + FP)

    # evaluation metrics
    #report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
    #print(pd.DataFrame(report).transpose().round(decimals=3))

    new_row.update({'accuracy': accuracy * 100,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'true_negative_rate': TNR[1] * 100,
                    'f2_score': f2 * 100,
                    'f3_score': f3 * 100})
    if update_res:
        res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True, axis=0) #res update
        return res
    else:
        return new_row

def update_avg_res(res, avg_res, condition, new_condition):
    for column in res.columns[res.columns.get_loc("accuracy"):]:  # every column from accuracy column

        avg_column = column + "_avg"
        std_column = column + "_std"

        avg_res.loc[new_condition, avg_column] = round(res.loc[condition, column].mean(), 2)
        avg_res.loc[new_condition, std_column] = round(res.loc[condition, column].std(), 2)

    return avg_res

def compute_avg_std(res, approach):
    """
    Given a dataset "res" that stores the metrics, it returns the avg and std for
    every metics stored in "res".
    """

    avg_res = pd.DataFrame()

    if approach == 1:
        for k in res["df"].unique():
            for j in res["model"].unique():
                condition = (res["df"] == k) & (res["model"] == j)

                new_row = {"df": k, "model": j}
                avg_res = pd.concat([avg_res, pd.DataFrame([new_row])], ignore_index=True, axis=0)  #avg_res update
                new_condition = (avg_res["df"] == k) & (avg_res["model"] == j)

                avg_res = update_avg_res(res = res,
                                         avg_res = avg_res,
                                         condition = condition,
                                         new_condition = new_condition)


    elif approach == 2:
        for k in res["df"].unique():
            for j in res["model"].unique():
                for h in res["set"].unique():
                    condition = (res["df"] == k) & (res["model"] == j) & (res["set"] == h)

                    new_row = {"df": k, "model": j, "set" : h}
                    avg_res = pd.concat([avg_res, pd.DataFrame([new_row])], ignore_index=True, axis=0)  # avg_res update
                    new_condition = (avg_res["df"] == k) & (avg_res["model"] == j) & (avg_res["set"] == h)

                    avg_res = update_avg_res(res, avg_res, condition, new_condition)

    return avg_res


"""## Rank-based metrics"""


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def arrange_predictions_and_targets(pred, target):
    # create dataframe that contain predictions and targets
    temp = pd.DataFrame()
    temp["prediction_value"] = pred
    temp["target"] = target

    # sort in descending order respect to the predictions
    temp.sort_values(by=['prediction_value'], inplace=True, ascending=False, ignore_index=True)

    # convert predictions from real value to 0 or 1
    temp.loc[temp["prediction_value"] >= 0.5, "prediction"] = 1
    temp.loc[temp["prediction_value"] < 0.5, "prediction"] = 0
    temp.drop('prediction_value', inplace=True, axis=1)

    # cast dataframe type to int
    temp = temp.astype(int)

    return temp


def getTpTnFpFn(df):
    TP = len(df.loc[df[["target", "prediction"]].eq(1).all(1)])
    TN = len(df.loc[df[["target", "prediction"]].eq(0).all(1)])
    FP = len(df.loc[(df["target"] == 0) & (df["prediction"] == 1)])
    FN = df.shape[0] - (TP + TN + FP)

    return TP, TN, FP, FN


def get_rank_at_k(df, total_relevant_docs, K=95):
    relevant_docs = 0

    for j in range(len(df)):

        if df.loc[j, "target"] == 1:
            relevant_docs = relevant_docs + 1

        recall_k = relevant_docs / total_relevant_docs

        if recall_k >= K / 100:
            return j

    print("Error")
    return 0


def update_results(pred_df, df, model, avg_res, approach, set_type=None, K=95):
    # compute TP, TN, FP, FN
    TP, TN, FP, FN = getTpTnFpFn(pred_df)
    retrieved_docs = TP + FP
    relevant_docs = TP + FN
    not_relevant_docs = TN + FP

    # EVALUATION MEASURES AT 95% RECALL

    # selection of the rows where recall@95
    last_positive_rank = get_rank_at_k(pred_df, relevant_docs)
    pred_df_95 = pred_df[:last_positive_rank + 1]
    pred_df_5 = pred_df[last_positive_rank + 1:]
    TP_95, TN_95, FP_95, FN_95 = getTpTnFpFn(pred_df_95)
    TP_5, TN_5, FP_5, FN_5 = getTpTnFpFn(pred_df_5)

    # TRUE NEGATIVE RATE AT 95% RECALL
    if not_relevant_docs == 0:
        print("Number of not relevant docs equals to 0")
        TNR_k = 0
    else:
        TNR_k = TN_5 / not_relevant_docs

    # PRECISION@95
    if retrieved_docs == 0:
        print("Number of retrieved docs equals to 0")
        precision_k = 0
    else:
        precision_k = TP_95 / retrieved_docs

    # WSS@95
    N = len(pred_df)
    WSS_k = ((N - last_positive_rank) / N) - ((100 - K) / 100)

    # update results dataset
    if approach == 1:
        condition = (avg_res["df"] == df) & (avg_res["model"] == model)
    elif approach == 2:
        condition = avg_res["set"] == set_type
    avg_res.loc[condition, "true_negative_rate@95"] = round(TNR_k * 100, 2)
    avg_res.loc[condition, "precision@95"] = round(precision_k * 100, 2)
    avg_res.loc[condition, "wss@95"] = round(WSS_k * 100, 2)

    # print("N:", N)
    # print("Last positive rank:", last_positive_rank)
    # print()

    return avg_res