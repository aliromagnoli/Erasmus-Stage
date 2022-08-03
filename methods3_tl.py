from transformers import AutoTokenizer
import pandas as pd

#model approach 3
model_ckpt = "distilbert-base-uncased" #da spostare in training_models

#create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

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