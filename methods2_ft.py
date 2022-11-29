import fasttext
import pandas as pd
import numpy as np
import methods_evaluation_metrics as eval
import methods_data_import_and_preprocessing as pr

N_EPOCHS = 20

def final_ft_preprocessing(train, test, sampling, seed, path):

    # OVERSAMPLING ON TRAINING SET
    X_ros, y_ros = pr.oversampling(train["text"].values.reshape(-1, 1), train["label"].values, sampling, seed)

    # creation of oversampled training set
    oversampled_train = pd.DataFrame(y_ros, columns=["label"])
    oversampled_train["text"] = X_ros
    oversampled_train["label"] = oversampled_train["label"].apply(lambda x: '__label__' + str(x))

    test = test.copy()
    test["label"] = test["label"].apply(lambda x: '__label__' + str(x))

    np.savetxt(path+"/train.txt", oversampled_train, fmt='%s')
    np.savetxt(path+"/test.txt", test[["label", "text"]], fmt='%s')

    return path+"/train.txt", path+"/test.txt"

def get_predictions(test, model):
    test_file = open(test, 'r')
    lines = test_file.readlines()
    true_label = []
    pred_label = []
    pred_value = []
    for line in lines:
        true_label.append(int(line.split()[0][-1]))
        pred_l, pred_v = model.predict(line.strip())
        pred_label.append(int(pred_l[0].split()[0][-1]))
        if int(pred_l[0].split()[0][-1]) == 0: #to always have the probability for the class 1
            pred_value.append(1-pred_v[0])
        else:
            pred_value.append(pred_v[0])

    metric_row = eval.evalmetrics(y_test=true_label,
                                  y_pred=pred_label)
    return metric_row, true_label, pred_label, pred_value

def ft_training(train, test, res_index, res, pred):

    row = {"df": res_index.get_df(),
           "model": "FT",
           "set": "test",
           "fold": res_index.get_fold(),
           "iteration": res_index.get_iter()}

    model = fasttext.train_supervised(input=train, lr=1.0, epoch=N_EPOCHS)
    metric_row, target, pred_label, pred_value = get_predictions(test, model)

    #update res
    m_row = row.copy()
    m_row.update(metric_row)
    res = pd.concat([res, pd.DataFrame([m_row])], ignore_index=True, axis=0)

    #update pred
    p_row = row.copy()
    p_row.update({"target": pd.Series(target), "pred": pd.Series(pred_value)})
    pred = pd.concat([pred, pd.DataFrame([p_row])], ignore_index=True, axis=0)

    return res, pred




