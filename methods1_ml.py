"""# Traditional ML approach"""

import methods_data_import_and_preprocessing as pr
import methods_evaluation_metrics as eval
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = {"SVM" : LinearSVC, "DT" : DecisionTreeClassifier, "RF" : RandomForestClassifier, "LR" : LogisticRegression}

def flatten_words(l, get_unique=False):
    """
    Given a list "l" containing strings,
    it returns the flatten version of the list,
    maintaining only the unique strings if get_unique=True.
    """
    qa = [s.split() for s in l]
    if get_unique:
        return sorted(list(set([w for sent in qa for w in sent])))
    else:
        return [w for sent in qa for w in sent]


def final_ml_preprocessing(train, test):
    """
    Given a training set "train" and a test set "test",
    it returns X_train, y_train, X_test, y_test, after applying
    a weighting scheme on the text and scaling the data.
    """
    import training_models as trm

    # splitting in X and y for both train and test
    X_train = train.loc[:, train.columns != 'label']
    y_train = train["label"]
    X_test = test.loc[:, test.columns != 'label']
    y_test = test["label"]

    if trm.SAMPLING is not None:
        X_train, y_train = pr.oversampling(X_train, y_train, trm.SAMPLING, trm.SEED[0])

    # resetting indexes for concat()
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # tf-idf
    all_text = X_train["text_clean"].values.tolist() + X_test["text_clean"].values.tolist()
    vocab = flatten_words(all_text, get_unique=True)
    tfidf = TfidfVectorizer(stop_words='english', vocabulary=vocab)
    training_matrix = tfidf.fit_transform(X_train["text_clean"])
    test_matrix = tfidf.fit_transform(X_test["text_clean"])

    # print("Training matrix:", training_matrix.shape)
    # print("Test matrix:", test_matrix.shape)

    X_train = pd.concat([X_train, pd.DataFrame(training_matrix.todense())], axis=1)  # add training_matrix to X_train
    X_test = pd.concat([X_test, pd.DataFrame(test_matrix.todense())], axis=1)

    # scaling data
    scaler = MinMaxScaler()
    features = list(
        set(X_train.columns) - set(["text", "text_clean", "label"]))  # all columns except text, text_clean and Label
    X_train = scaler.fit_transform(X_train[features].values)
    y_train = y_train.values
    X_test = scaler.transform(X_test[features].values)

    return X_train, y_train, X_test, y_test


def ml_training(X_train, y_train, X_test, y_test, res_index):
    """
    Given a dataset "pred" to store the predictions in, a model function "model",
    a int "seed", a dictionary "row" with all the information to update "pred",
    "X_train", "y_train" and "X_test", it trains the model, computes the predictions
    on the test set and updates "pred".
    """
    import training_models as trm

    print(res_index.get_fold())
    for model_name in models:
        new_row = {"df": res_index.get_df(),
                   "model": model_name,
                   "fold": res_index.get_fold(),
                   "iteration": res_index.get_iter()}

        if model_name == "RF":
            m = models[model_name](class_weight="balanced", random_state=trm.SEED[0], n_estimators=50, max_depth=8)
        elif model_name == "SVM":
            m = models[model_name](class_weight="balanced", random_state=trm.SEED[0])  # , C=0.5, max_iter=2000)
            m = CalibratedClassifierCV(m)
        else:
            m = models[model_name](class_weight="balanced", random_state=trm.SEED[0])

        m = m.fit(X_train, y_train)

        # model predictions on the test set

        pred_class_test = m.predict(X_test)  # predicted classes
        pred_test = m.predict_proba(X_test)  # predicted probabilities
        pred_test = pred_test[:, 1]

        pred_row = new_row.copy()
        pred_row["target"] = y_test
        pred_row["pred"] = pd.DataFrame(pred_test)
        trm.pred = pd.concat([trm.pred, pd.DataFrame([pred_row])], ignore_index=True, axis=0) # pred update

        # evaluation metrics

        print("\n", new_row["model"], "RESULTS:\n")
        trm.res = eval.evalmetrics(trm.res, y_test, pred_class_test, m.classes_, new_row)

