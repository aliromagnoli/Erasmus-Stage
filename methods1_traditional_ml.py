"""# Traditional ML approach"""

import methods_data_import_and_preprocessing
import methods_evaluation_metrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV


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


def final_ml_preprocessing(train, test, seed, sampling=None):
    """
    Given a training set "train" and a test set "test",
    it returns X_train, y_train, X_test, y_test, after applying
    a weighting scheme on the text and scaling the data.
    """

    # splitting in X and y for both train and test
    X_train = train.loc[:, train.columns != 'label']
    y_train = train["label"]
    X_test = test.loc[:, test.columns != 'label']
    y_test = test["label"]

    if sampling is not None:
        X_train, y_train = methods_data_import_and_preprocessing.oversampling(X_train, y_train, sampling, seed)

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
        set(X_train.columns) - set(["text", "text_clean", "Label"]))  # all columns except text, text_clean and Label
    X_train = scaler.fit_transform(X_train[features].values)
    y_train = y_train.values
    X_test = scaler.transform(X_test[features].values)

    return X_train, y_train, X_test, y_test


def ml_training(pred, model, seed, row, X_train, y_train, X_test, y_test, res):
    """
    Given a dataset "pred" to store the predictions in, a model function "model",
    a int "seed", a dictionary "row" with all the information to update "pred",
    "X_train", "y_train" and "X_test", it trains the model, computes the predictions
    on the test set and updates "pred".
    """

    if row["model"] == "RF":
        m = model(class_weight="balanced", random_state=seed, n_estimators=50,
                  max_depth=8)  # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
    elif row["model"] == "SVM":
        m = model(class_weight="balanced", random_state=seed)  # , C=0.5, max_iter=2000)
        m = CalibratedClassifierCV(m)
    else:
        m = model(class_weight="balanced", random_state=seed)

    m = m.fit(X_train, y_train)

    # model predictions on the test set

    pred_class_test = m.predict(X_test)  # predicted classes
    pred_test = m.predict_proba(X_test)  # predicted probabilities
    pred_test = pred_test[:, 1]

    pred_row = row.copy()
    pred_row["target"] = y_test
    pred_row["pred"] = pd.DataFrame(pred_test)
    pred = pd.concat([pred, pd.DataFrame([pred_row])], ignore_index=True, axis=0) # pred update

    # evaluation metrics

    print("\n", row["model"], "RESULTS:\n")
    res = methods_evaluation_metrics.evalmetrics(res, y_test, pred_class_test, m.classes_, row)

    return pred, res
