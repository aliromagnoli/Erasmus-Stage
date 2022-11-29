import methods_data_import_and_preprocessing as pr
import methods_evaluation_metrics as eval
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#PARAMETERS

TEST_SIZE = 0.8
MODEL_CKPT = "distilbert-base-uncased"
MODEL_NAME = f"{MODEL_CKPT}-approach3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

#create tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)


def final_tl_preprocessing(train, test, sampling, sorting, seed) -> object:

    if sorting:
        oversampled_train = train
    else:
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
        "train": datasets.Dataset.from_pandas(oversampled_train.reset_index(drop=True)),
        "validation": datasets.Dataset.from_pandas(valid.reset_index(drop=True)),
        "test": datasets.Dataset.from_pandas(test.reset_index(drop=True))})
    
    return tokenize_whole_df(df=df)


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

def tl_training(df, epochs_res, res_index, res, pred):

    model = (AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT).to(DEVICE)) #num_labels=2
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
                                      log_level="error",
                                      group_by_length=True)

    row = {"df": res_index.get_df(),
           "model": "DISTILBERT",
           "fold": res_index.get_fold(),
           "iteration": res_index.get_iter()}

    #training + validation

    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=eval.tl_metrics,
                      train_dataset=df["train"],
                      eval_dataset=df["validation"],
                      tokenizer=tokenizer)
    # trainer._get_train_sampler()
    batches_distribution(trainer)
    trainer.train()

    count = 0
    best_row = {}

    for i in trainer.state.log_history:
        if "eval_loss" in i:

            new_row = row.copy()
            new_row.update({'set': "validation", #metrics
                            'epoch': i["epoch"],
                            'loss': i["eval_loss"],
                            'accuracy': i["eval_accuracy"],
                            'precision': i["eval_precision"],
                            'recall': i["eval_recall"],
                            'true_negative_rate': i["eval_true_negative_rate"],
                            'f2_score': i["eval_f2_score"],
                            'f3_score': i["eval_f3_score"],
                            'target': i["eval_pred_df"]["target"],
                            'pred': i["eval_pred_df"]["prediction"],
                            'fp': i["eval_fp"],
                            'fn': i["eval_fn"],
                            'tp': i["eval_tp"],
                            'tn': i["eval_tn"],
                            'fp_ratio': i["eval_fp_ratio"],
                            'fn_ratio': i["eval_fn_ratio"],
                            'tp_ratio': i["eval_tp_ratio"],
                            'tn_ratio': i["eval_tn_ratio"]})

            if count == 0:
                best_row = new_row.copy()
            count = count + 1

            epochs_res = pd.concat([epochs_res, pd.DataFrame([new_row])], ignore_index=True, axis=0) #save validation metrics for each epoch

            if new_row["recall"] > best_row["recall"]:
                best_row = new_row.copy()
            elif (new_row["recall"] == best_row["recall"]) and (new_row["precision"] > best_row["precision"]):
                best_row = new_row.copy()

    #test evaluation
    preds_output = trainer.predict(df["test"])
    new_row = row.copy()
    new_row.update({'set': "test",  # metrics
                    'epoch': best_row["epoch"],
                    'loss': preds_output.metrics["test_loss"],
                    'accuracy': preds_output.metrics["test_accuracy"],
                    'precision': preds_output.metrics["test_precision"],
                    'recall': preds_output.metrics["test_recall"],
                    'true_negative_rate': preds_output.metrics["test_true_negative_rate"],
                    'f2_score': preds_output.metrics["test_f2_score"],
                    'f3_score': preds_output.metrics["test_f3_score"],
                    'target' : preds_output.metrics["test_pred_df"]["target"],
                    'pred' : preds_output.metrics["test_pred_df"]["prediction"],
                    'fp': preds_output.metrics["test_fp"],
                    'fn': preds_output.metrics["test_fn"],
                    'tp': preds_output.metrics["test_tp"],
                    'tn': preds_output.metrics["test_tn"],
                    'fp_ratio': preds_output.metrics["test_fp_ratio"],
                    'fn_ratio': preds_output.metrics["test_fn_ratio"],
                    'tp_ratio': preds_output.metrics["test_tp_ratio"],
                    'tn_ratio': preds_output.metrics["test_tn_ratio"]})

    epochs_res = pd.concat([epochs_res, pd.DataFrame([new_row])], ignore_index=True, axis=0) #save test metrics for each epoch
    epochs_res.drop(['target', 'pred'], axis=1, inplace=True) #remove targets and preds from metrics results for each epoch

    res = pd.concat([res, pd.DataFrame([best_row])], ignore_index=True, axis=0)  #save validation metrics of the best epoch
    res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True, axis=0)  # save test metrics of the best epoch

    pred = pd.concat([pred, res.loc[~res['target'].isnull(), ["df", "model", "set", "fold", "iteration", "target", "pred"]].copy()],
                        ignore_index=True, axis=0)  #pred contains targets and preds for best epochs
    res.drop(['target', 'pred'], axis=1, inplace=True)  #remove targets and preds from metrics results for best epochs

    return epochs_res, res, pred


def batches_distribution(trainer):
    batch = {}
    ii = 0
    for b in trainer.get_train_dataloader():
      batch[ii] = b
      ii = ii + 1

    count = 0
    for b in range(len(batch)):
      batch[b]["#_positive"] = sum(batch[b]["labels"]).numpy().item()
      if batch[b]["#_positive"] < 7:
        count = count + 1

    batch_df = pd.DataFrame.from_dict(batch, orient='index')
    print(batch_df["#_positive"].unique())

    plt.hist(batch_df["#_positive"], bins=np.arange(15)-0.5)
    plt.xlabel('number of positive examples per batch')
    plt.ylabel('frequency')
    plt.xticks(range(15))
    plt.xlim([0, 15])
    plt.savefig("positives_in_batches.png")
    plt.show()
    #files.download("positives_in_batches.png")










