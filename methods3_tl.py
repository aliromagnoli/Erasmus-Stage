import methods_data_import_and_preprocessing as pr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

#PARAMETERS
TEST_SIZE = 0.8
MODEL_CKPT = "distilbert-base-uncased"
MODEL_NAME = f"{MODEL_CKPT}-approach3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10 #64
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5

#create tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)


def final_tl_preprocessing(train, test, sampling, seed) -> object:

    # OVERSAMPLING ON TRAINING SET
    X_ros, y_ros = pr.oversampling(train["text"].values.reshape(-1, 1), train["label"].values, sampling, seed)

    # creation of oversampled training set
    oversampled_train = pd.DataFrame(X_ros, columns=["text"])
    oversampled_train["label"] = y_ros

    # test-validation split
    test, valid = train_test_split(test,
                                   train_size=TEST_SIZE,
                                   random_state=random.seed(seed),
                                   shuffle=True,
                                   stratify=test['label'])

    df = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train, preserve_index=False),
        "validation": datasets.Dataset.from_pandas(valid, preserve_index=False),
        "test": datasets.Dataset.from_pandas(test, preserve_index=False)})
    
    return tokenize_whole_df(df = df)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def tokenize_whole_df(df):
    """
    #print token id of special tokens
    tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
    data = sorted(tokens2ids, key=lambda x: x[-1])
    tokens_id = pd.DataFrame(data, columns=["Special Token", "Special Token ID"])
    print(tokens_id.T)
    """
    #tokenize the dataset
    return df.map(tokenize, batched=True, batch_size=None)

def tl_training(df):
    model = (AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=2).to(DEVICE))
    logging_steps = len(df["train"]) // BATCH_SIZE

    training_args = TrainingArguments(output_dir=MODEL_NAME,
                                      num_train_epochs=NUM_EPOCHS,
                                      learning_rate=LEARNING_RATE,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=True,
                                      log_level="error")
    """
    row = {"df": dataset_name,
           "model": "DISTILBERT",
           "set": "train",
           "fold": res_index.get_fold(),
           "iteration": res_index.get_iter(),
           "epoch": epoch,
           "loss": epoch_loss}
    """

    trainer = Trainer(model=model, args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=df["train"],
                      eval_dataset=df["validation"],
                      tokenizer=tokenizer)

    trainer.train();
    preds_output = trainer.predict(df["validation"])
    print(preds_output.metrics)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    y_valid = np.array(df["validation"]["label"])
    plot_confusion_matrix(y_preds, y_valid)


def plot_confusion_matrix(y_preds, y_true):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}