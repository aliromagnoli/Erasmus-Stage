import pandas as pd
from nltk.tokenize import regexp_tokenize
import re
from imblearn.over_sampling import RandomOverSampler
import spacy
nlp = spacy.load('en_core_web_sm')
all_stopwords = nlp.Defaults.stop_words

"""
# Data Import

##Import of data
"""


def print_value_counts(l, df):
    """
    Given a list of column names "l" and a dataset "df",
    it prints the value counts for each specified variable.
    """
    for item in l:
        print(df[item].value_counts(), "\n")


def rename_columns(names, df):
    """
    Given a dictionary "names" like {"old_name" : "new_name"} and a dataset "df",
    it renames the column names specified as "old_name" with new names specified as "new_name".
    """
    for key in names:
        df.rename(columns={key: names[key]}, inplace=True)


"""
## Column preprocessing

### Text preprocessing
"""


def listToString(s):
    """
    Given a list "s", it returns "s" as a string.
    """
    str1 = ""
    for ele in s:
        str1 = str1 + " " + ele
    return str1


def split_on_word(text):
    """
    Given a list or string "text", it uses a regular expression tokenizer
    (that keeps apostrophes) to return a tokenized list of lists
    (one list for each sentence: [[word, word], [word, word, ..., word], ...]).
    """
    if type(text) is list:
        return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
    else:
        return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")


def count_words(df, column):
    """
    Given a dataset "df" and a column name "column",
    it returns the dataset "df" with a new column containing the number of words in "column".
    """
    df = df.copy()
    col_name = "n_words_in_" + column
    df[col_name] = df[column].apply(lambda x: len(split_on_word(x)))
    return df


def clean_text(df, col):
    """
    Given a dataset "df" and a column name "col",
    it modifies the column "col", keeping only alpha-numeric characters and
    replacing all white space with a single space, and then return "df"
    """
    df = df.copy()

    # [^A-Za-z0-9]+: regex to match a string of characters that are not a letters or numbers
    return df[col].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', str(x).lower())) \
        .apply(lambda x: re.sub('\s+', ' ', x).strip())


"""### Features preprocessing"""


def from_list_of_values_to_columns(col, df, nolist=False, print_count=False):
    """
    Given a column name "col" and a dataset "df",
    it converts a column containing lists of values to a binary column for each value.
    """
    df = df.copy()

    # obtaining the unique values
    if nolist:
        for index, content in enumerate(df[col]):
            df.at[index, col] = list(content.split(","))
    else:
        df[col] = df[col].apply(eval)

    col_dict = {}
    for i in df[col]:  # obtain value_count in a dictionary
        for j in i:
            if j not in col_dict:
                col_dict[j] = 1  # new column
            else:
                col_dict[j] += 1  # update column count

    series = pd.Series([x for _list in df[col] for x in _list])  # reducing its dimensions from 2 to 1

    if print_count:
        print(series.value_counts())  # display value count

    # creating new binary columns

    bool_dict = {}  # create boolean dict (the binary value for every colum in col_dict and for every row in the df)
    for i, item in enumerate(col_dict.keys()):
        bool_dict[item] = df[col].apply(lambda x: item in x)

    return pd.DataFrame(bool_dict).astype(int)


"""## Update `col_names`"""


def update_col_names(col_names, col, name_df, sub_features):
    """
    Given a dataset to update "col_names", a column name "col",
    a dataset name "name_df" and a list of values "sub_features",
    it update the dataset "col_names" with "sub_features"
    """

    #col_names = col_names.copy()
    # useful transformation for assigning a list to a dataframe cell
    l = col_names.index[col_names["feature"] == col].tolist()
    col_names.at[l[0], name_df] = sub_features

    return col_names


"""## Topic search"""


def find_documents_about_topic(df, column, new_column, l):
    """
    Given a dataset "df", a column name "column", a new column name "new_column" and a list of strings "l",
    it modifies "df" adding a binary column that specify for every row if "column" contains at least 1 of the strings in "l".
    """

    x = df[column][df[column].str.contains('|'.join(l))]  # rows in df[column] that contains at least 1 item of "l"

    df[new_column] = 0
    df.loc[df.index.isin(x.index), new_column] = 1  # assigning 1 to the corresponding rows of x in df

    print("Number of documents that", new_column, ":", len(x))

    return df

def topic_search(df, l1, l2, text_clean=True):
    """
    Given a dataset "df" and two lists of strings "l1" and "l2",
    it perform the steps to modifies "df" adding a binary column that specify for every row if "column" contains at least 1 of the strings in "l".
    """
    df = find_documents_about_topic(df, "text_clean", "contains_topic", l1)
    df = find_documents_about_topic(df, "text_clean", "contains_other_topic", l2)

    print("\nChecking how many documents don't contain any of the searched words:\n")
    print(df[["contains_topic", "contains_other_topic"]].eq(0).all(1).value_counts(), "\n")
    print("True -> they don't contain any of the words")
    print("False -> they contain at least one word")

    print("\nChecking how many documents contain at least one word from each list:\n")
    print(df[["contains_topic", "contains_other_topic"]].eq(1).all(1).value_counts(), "\n")
    print("True -> they contain words from both lists")
    print("False -> they don't contain words from both lists\n")

    return df

"""## Final preprocessing"""

def oversampling(X_train, y_train, sampling, seed):

    # oversampling on training set
    ros = RandomOverSampler(sampling_strategy=sampling, random_state=seed)
    # resampling X, y
    X_train, y_train = ros.fit_resample(X_train, y_train)
    return X_train, y_train


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


def additional_preprocessing(df):

    for index, text in enumerate(df["text"]):
        doc = nlp(text)
        l = []
        # lemmatization
        for token in doc:
            l.append(token.lemma_)
        #stopwords removal
        l = [word for word in l if not word in nlp.Defaults.stop_words]
        final_text = " ".join(map(str, l))
        #print(final_text)
        df.at[index, "text"] = final_text

    return df

def maintain_only_text_label(df, cleantext, paraphrasing, approach):

    if cleantext:
        df.drop(["text"], axis=1, inplace=True)
        rename_columns({"text_clean": "text"}, df)
    else:
        df.drop(["text_clean"], axis=1, inplace=True)

    if paraphrasing:
        cols = ["label", "text", "paraphrased_text"]
    else:
        cols = ["label", "text"]

    if approach != "ML":
        df.drop(labels=df.columns.difference(cols), axis=1, inplace=True) #remove everything except cols
    else:
        if not paraphrasing:
            df.drop(labels=["paraphrased_text"], axis=1, inplace=True)

    print(df.shape)
    return df  # dataset with "label and text"

def sorting_for_training(df, batch_size):

    positive_df = df[df["label"] == 1].copy()
    negative_df = df[df["label"] == 0].copy()

    n_pos = round(batch_size/2)
    n_neg = round(batch_size/2)

    if (batch_size % 2) != 0:
      n_neg = n_neg + 1

    if len(positive_df) < n_pos:
      n_pos = len(positive_df)
      n_neg = batch_size - n_pos

    n_batches = round(len(negative_df)/n_neg)

    if ((len(negative_df)/n_neg) % 2) != 0:
      n_batches = n_batches + 1

    sorted_df = pd.DataFrame()

    for index in range(n_batches):

        pos = positive_df.sample(n=n_pos)

        neg = negative_df[:n_neg]
        negative_df.drop(index=negative_df.index[:n_neg], axis=0, inplace=True)

        batch = pd.concat([pos, neg], ignore_index=True)
        batch = batch.sample(frac=1).reset_index(drop=True)

        sorted_df = pd.concat([sorted_df, batch], ignore_index=True)

    return sorted_df

def perform_options(train, test, paraphrasing, text_preprocessing, sorting, cleantext, batch_size):

    if paraphrasing:
        new_rows = train[train.columns.difference(["text"])].copy() #every column except text
        rename_columns({"paraphrased_text": "text"}, new_rows)
        new_rows = new_rows.dropna().reset_index(drop=True)
        if cleantext:
            new_rows["text"] = clean_text(new_rows, "text")
        train.drop(labels="paraphrased_text", axis=1, inplace=True)
        train = pd.concat([train, new_rows], ignore_index=True)
        test.drop(labels="paraphrased_text", axis=1, inplace=True)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    if text_preprocessing:
        train = additional_preprocessing(train)
        test = additional_preprocessing(test)

    if sorting:
        train = sorting_for_training(train, batch_size)

    return train, test