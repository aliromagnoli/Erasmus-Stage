# -*- coding: utf-8 -*-
"""data_import.py

Datasets are imported from: https://www.dropbox.com/sh/ud5sf1fy6m7o219/AAD9pkY5gYe_XYV2oHDw68uva?dl=0
and from: https://github.com/hannousse/Semantic-Scholar-Evaluation

"""


"""## Libraries and methods"""

#libraries
import methods_data_import_and_preprocessing as pr
import os
import numpy as np
import pandas as pd
from IPython.display import display
import nltk
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

"""## Import of data"""

path = os.getcwd() + "\datasets\original datasets"
dataset = dict()

old_datasets = ["ace", "copd", "ppi"]
new_datasets = ["alhammad", "ghasemi", "goulao", "guinea", "santos", "shahin", "yang"]

"""### ACE Inhibitors dataset"""

#import of data
dataset["ace"] = pd.read_csv(path + "/ACEInhibitors.tsv", sep='\t')
print(dataset["ace"].shape)
display(dataset["ace"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["ace"].columns)

#value counts
variables = ["MH", "STAT", "VI", "IP", "DP", "FAU", "AU", "AD", "LA", "PT", "PL", "TA", "JT"]
pr.print_value_counts(l=variables, df=dataset["ace"])

#target distribution
pr.print_value_counts(["Label"], dataset["ace"])

#renaming columns
names = {"DP" : "publication_date", "FAU" : "full_authors", "PT" : "publication_type", "PL" : "publication_place", "TA" : "journal_title_abbreviation", "MH" : "mesh_terms", "Label" : "label"}
pr.rename_columns(names=names, df=dataset["ace"])

#column selection
selected_columns = ["publication_date", "full_authors", "publication_type", "publication_place", "journal_title_abbreviation", "Title", "Abstract", "mesh_terms", "label"]
dataset["ace"] = dataset["ace"][selected_columns]

"""### Chronic obstructive pulmonary disease (COPD) dataset"""

#import of copd dataset
dataset["copd"] = pd.read_csv(path + "/copd.tsv", sep='\t')
print(dataset["copd"].shape)
display(dataset["copd"].head(3))

#check if Title is identical to Abstract
print(all(dataset["copd"]["Title"] == dataset["copd"]["Abstract"]))
dataset["copd"].drop('Title', inplace=True, axis=1) #drop Title

#target distribution
pr.print_value_counts(["Label"], dataset["copd"])
pr.rename_columns(names={"Label" : "label"}, df=dataset["copd"])

"""### Proton Pump Inhibitors dataset"""

#import of ppi dataset
dataset["ppi"] = pd.read_csv(path + "/ProtonPumpInhibitors.tsv", sep='\t')
print(dataset["ppi"].shape)
display(dataset["ppi"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["ppi"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["ppi"])

#target distribution
pr.print_value_counts(["Label"], dataset["ppi"])

#renaming columns
pr.rename_columns(names=names, df=dataset["ppi"])

#column selection
dataset["ppi"] = dataset["ppi"][selected_columns]

"""### Alhammad 2018 dataset"""

#import of data
df0 = pd.read_csv(path + "/alhammad-2018-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/alhammad-2018-included.csv")
df1["label"] = 1
dataset["alhammad"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Alhammad dataset shape:", dataset["alhammad"].shape)
display(dataset["alhammad"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["alhammad"].columns)

#value counts
variables = ["Publication year", "Authors", "Item type", "Journal", "Keywords"]
pr.print_value_counts(l=variables, df=dataset["alhammad"])

#target distribution
pr.print_value_counts(["label"], dataset["alhammad"])

#renaming columns
names = {"Publication year" : "publication_date", "Authors" : "full_authors", "Item type" : "publication_type", "Keywords" : "mesh_terms"}
pr.rename_columns(names=names, df=dataset["alhammad"])

#column selection
selected_columns2 = ["publication_date", "full_authors", "publication_type", "Title", "Abstract", "mesh_terms", "label"]
dataset["alhammad"] = dataset["alhammad"][selected_columns2]


"""### Ghasemi 2019 dataset"""

#import of data
df0 = pd.read_csv(path + "/ghasemi-2019-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/ghasemi-2019-included.csv")
df1["label"] = 1
dataset["ghasemi"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Ghasemi dataset shape:", dataset["ghasemi"].shape)
display(dataset["ghasemi"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["ghasemi"].columns)

#value counts
variables = ["Publication year", "Authors", "Item type", "Journal", "Keywords", "Address"]
pr.print_value_counts(l=variables, df=dataset["ghasemi"])

#target distribution
pr.print_value_counts(["label"], dataset["ghasemi"])

#renaming columns
pr.rename_columns(names=names, df=dataset["ghasemi"])

#column selection
dataset["ghasemi"] = dataset["ghasemi"][selected_columns2]


"""### Goulao 2016 dataset"""

#import of data
df0 = pd.read_csv(path + "/goulao-2016-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/goulao-2016-included.csv")
df1["label"] = 1
dataset["goulao"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Goulao dataset shape:", dataset["goulao"].shape)
display(dataset["goulao"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["goulao"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["goulao"])

#target distribution
pr.print_value_counts(["label"], dataset["goulao"])

#renaming columns
pr.rename_columns(names=names, df=dataset["goulao"])

#column selection
dataset["goulao"] = dataset["goulao"][selected_columns2]


"""### Guinea 2016 dataset"""

#import of data
df0 = pd.read_csv(path + "/guinea-2016-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/guinea-2016-included.csv")
df1["label"] = 1
dataset["guinea"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Guinea dataset shape:", dataset["guinea"].shape)
display(dataset["guinea"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["guinea"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["guinea"])

#target distribution
pr.print_value_counts(["label"], dataset["guinea"])

#renaming columns
pr.rename_columns(names=names, df=dataset["guinea"])

#column selection
dataset["guinea"] = dataset["guinea"][selected_columns2]


"""### Santos 2018 dataset"""

#import of data
df0 = pd.read_csv(path + "/santos-2018-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/santos-2018-included.csv")
df1["label"] = 1
dataset["santos"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Santos dataset shape:", dataset["santos"].shape)
display(dataset["santos"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["santos"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["santos"])

#target distribution
pr.print_value_counts(["label"], dataset["santos"])

#renaming columns
pr.rename_columns(names=names, df=dataset["santos"])

#column selection
dataset["santos"] = dataset["santos"][selected_columns2]


"""### Shahin 2017 dataset"""

#import of data
df0 = pd.read_csv(path + "/shahin-2017-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/shahin-2017-included.csv")
df1["label"] = 1
dataset["shahin"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Shahin dataset shape:", dataset["shahin"].shape)
display(dataset["shahin"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["shahin"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["shahin"])

#target distribution
pr.print_value_counts(["label"], dataset["shahin"])

#renaming columns
pr.rename_columns(names=names, df=dataset["shahin"])

#column selection
dataset["shahin"] = dataset["shahin"][selected_columns2]


"""### Yang 2016 dataset"""

#import of data
df0 = pd.read_csv(path + "/yang-2016-excluded.csv")
df0["label"] = 0
df1 = pd.read_csv(path + "/yang-2016-included.csv")
df1["label"] = 1
dataset["yang"] = pd.concat([df0, df1], axis=0, ignore_index=True)

print("Yang dataset shape:", dataset["yang"].shape)
display(dataset["yang"].head(3))

"""#### Columns distribution and selection"""

#columns
print(dataset["yang"].columns)

#value counts
pr.print_value_counts(l=variables, df=dataset["yang"])

#target distribution
pr.print_value_counts(["label"], dataset["yang"])

#renaming columns
pr.rename_columns(names=names, df=dataset["yang"])

#column selection
dataset["yang"] = dataset["yang"][selected_columns2]


"""## Columns preprocessing

### Text preprocessig

- Checking missing values, 
- Concatenating `Title`, `Abstract` and `mesh_terms` in `text`,
- Preprocessing the text in `text_clean`,
- Removing missing values.
"""

## Variable mesh_terms

for i in dataset:

  if i != "copd":

    #replace missing values with empty list
    dataset[i]["mesh_terms"] = dataset[i]["mesh_terms"].replace(np.NaN, "[]")

    #from list of strings to string
    if i not in new_datasets: #in these datasets they are already strings
      for index, value in enumerate(dataset[i]["mesh_terms"]):
        dataset[i].loc[index, "mesh_terms"] = pr.listToString(eval(value))

    dataset[i]['mesh_terms'] = dataset[i]['mesh_terms'].str.replace('[/*]',' ', regex=True)
    dataset[i]['mesh_terms'] = dataset[i]['mesh_terms'].str.replace(';', ', ')
    dataset[i] = pr.count_words(dataset[i], "mesh_terms")

  ## Variables Title ad Abstract

  #count number of words in Title
    dataset[i] = pr.count_words(dataset[i], "Title")

  #count number of words in Abstract
  dataset[i]["Abstract"] = dataset[i]["Abstract"].replace(np.NaN, "[]") #replace missing values with empty list
  dataset[i]["Abstract"] = dataset[i]["Abstract"].astype(str)
  dataset[i] = pr.count_words(dataset[i], "Abstract")

  #checking missing values
  print("Dataset", i, "shape:", dataset[i].shape)
  print(dataset[i].isnull().sum(axis=0), "\n")

  #concatenating Title, Abstract (and mesh_terms)
  if i != "copd":
    dataset[i]['text'] = dataset[i]['Title'] + " " + (dataset[i]['Abstract']).fillna(' ') + " " + (dataset[i]['mesh_terms']).fillna(' ')
    dataset[i].drop(['Title', "Abstract", "mesh_terms"], inplace=True, axis=1)
  else:
    dataset[i]['text'] = (dataset[i]['Abstract']).fillna(' ')
    dataset[i].drop("Abstract", inplace=True, axis=1)

  ## Variable text_clean

  #text preprocessing
  dataset[i]['text_clean'] = pr.clean_text(dataset[i], 'text')
  dataset[i] = pr.count_words(dataset[i], "text_clean")

  #removing Nan and checking missing values again
  #print("Missing values after creating \"text\":")
  #print(dataset[i].isnull().sum(axis=0))
  dataset[i] = dataset[i].dropna()
  dataset[i] = dataset[i].reset_index()
  print("New dataset shape:", dataset[i].shape, "\n")

"""### Features preprocessing"""

#creating col_names (used to store columns' names)
cols = ["feature"]
cols.extend(old_datasets)
cols.extend(new_datasets)
col_names = pd.DataFrame(columns = cols)
new_features = ["contains_topic","contains_other_topic","n_words_in_mesh_terms",
                        "n_words_in_Title","n_words_in_Abstract","n_words_in_text_clean"]
selected_columns.extend(new_features)
col_names["feature"] = list(set(selected_columns) - {'Title', 'Abstract', 'mesh_terms', 'label'})
print("\n\n", col_names["feature"], "\n\n")

for i in dataset:

  if i != "copd":

    ## Variable publication_date

    #removing the day and maintaining only year and month
    for index, content in enumerate(dataset[i]["publication_date"]):
      if len(nltk.word_tokenize(str(content))) > 2:
        dataset[i].loc[index, "publication_date"] = " ".join(nltk.word_tokenize(str(content))[0:2])
    #print(dataset[i]["publication_date"].value_counts(), "\n")


    ## Variable publication_type
    nolist = False
    if i in new_datasets:
      nolist = True
    temp = pr.from_list_of_values_to_columns("publication_type", dataset[i], nolist=nolist)
    col_names = pr.update_col_names(col_names, "publication_type", i, list(temp.columns))
    dataset[i].drop("publication_type", inplace=True, axis=1)
    dataset[i] = dataset[i].join(temp)


    ## Variable full_authors
    temp = pr.from_list_of_values_to_columns("full_authors", dataset[i], nolist=nolist)
    col_names = pr.update_col_names(col_names, "full_authors", i, list(temp.columns))
    dataset[i].drop("full_authors", inplace=True, axis=1)
    dataset[i] = dataset[i].join(temp)

"""## Topic search

### Ace inhibitors related documents
"""

list1 = ["ace", "angiotensin converting enzyme"]
list2 = ["alacepril", "captopril", "zofenopril", "enalapril", "ramipril",
          "quinapril", "perindopril", "lisinopril", "benazepril", "imidapril",
          "trandolapril", "cilazapril", "fosinopril", "moexipril"]

dataset["ace"] = pr.find_documents_about_topic(dataset["ace"], "text_clean", "contains_topic", list1)
dataset["ace"] = pr.find_documents_about_topic(dataset["ace"], "text_clean", "contains_other_topic", list2)

print("Checking how many documents don't contain any of the searched words:\n")
print(dataset["ace"][["contains_topic", "contains_other_topic"]].eq(0).all(1).value_counts(), "\n")
print("True -> they don't contain any of the words")
print("False -> they contain at list one word")

"""### COPD related documents"""

list1 = ["chronic obstructive pulmonary disease", "copd"]
list2 = ["chronic obstructive lung disease", "cold", "chronic obstructive airway disease", "coad"]

dataset["copd"] = pr.find_documents_about_topic(dataset["copd"], "text_clean", "contains_topic", list1)
dataset["copd"] = pr.find_documents_about_topic(dataset["copd"], "text_clean", "contains_other_topic", list2)

print("Checking how many documents don't contain any of the searched words:\n")
print(dataset["copd"][["contains_topic", "contains_other_topic"]].eq(0).all(1).value_counts(), "\n")
print("True -> they don't contain any of the words")
print("False -> they contain at list one word")

"""### PPI related documents"""

list1 = ["proton pump inhibitors", "ppi", "ppis"]
list2 = ["omeprazole", "lansoprazole", "dexlansoprazole", "esomeprazole", "pantoprazole", "rabeprazole", "ilaprazole"]

dataset["ppi"] = pr.find_documents_about_topic(dataset["ppi"], "text_clean", "contains_topic", list1)
dataset["ppi"] = pr.find_documents_about_topic(dataset["ppi"], "text_clean", "contains_other_topic", list2)

print("Checking how many documents don't contain any of the searched words:\n")
print(dataset["ppi"][["contains_topic", "contains_other_topic"]].eq(0).all(1).value_counts(), "\n")
print("True -> they don't contain any of the words")
print("False -> they contain at list one word")

#updating col_names
for j in new_features:
  for k in dataset:
    if not (((k == "copd") and (j in ["n_words_in_Title", "n_words_in_mesh_terms"])) or
            ((k in new_datasets) and (j in ["contains_topic", "contains_other_topic"]))):
      col_names.loc[col_names["feature"]==j, k] = [[j]]

"""## Final data preprocessing

#### One hot encode for categorical variables
"""

#one hot encode for categorical variables

enc = OneHotEncoder()

for i in dataset:

  if i != "copd": #doesn't have any categorical variable

    if i in new_datasets:
      cat_features = ['publication_date']
    else:
      cat_features = ['publication_date', 'publication_place', 'journal_title_abbreviation']  # categorical features

    enc_df = pd.DataFrame(enc.fit_transform(dataset[i][cat_features]).toarray()) #one hot encode df for categorical features
    enc_df.columns = enc.get_feature_names_out(cat_features) #renaming columns of enc_df
    print("One hot encode dataset shape:", enc_df.shape)

    dataset[i] = dataset[i].join(enc_df)
    dataset[i] = dataset[i].drop(cat_features, axis=1)

    #updating col_names
    for j in cat_features:
      pr.update_col_names(col_names, j, i, [x for x in enc_df.columns if x.startswith(j)])

  dataset[i] = dataset[i].drop("index", axis=1)
  print("Final", i, "shape:", dataset[i].shape, "\n")

del enc_df

path = os.getcwd() + "/datasets/preprocessed datasets"

#saving the preprocessed datasets and col_names
cols.remove("feature")
for df in cols:
  with open(path + "/preprocessed_" + df + "_no_feature_selection.csv", 'w', encoding = 'utf-8-sig') as f:
    dataset[df].to_csv(f)
with open(path + "/columns_names.csv", 'w', encoding = 'utf-8-sig') as f:
    col_names.to_csv(f)

"""### Feature importance and feature selection"""

# creating the random forest that computes feature importance

for i in dataset:

  forest = RandomForestClassifier(random_state=0)
  forest.fit(dataset[i][dataset[i].columns.difference(["text", "text_clean", "label"])], dataset[i]["label"])

  #computing feature importance
  importances = forest.feature_importances_

  features = list(set(dataset[i].columns) - {"text", "text_clean", "label"}) #all columns except text, text_clean and label

  forest_importances = pd.Series(importances, index=features)
  forest_importances = forest_importances.sort_values(ascending=False)

  print("\nInitial number of features:", len(forest_importances))

  #non-important features
  non_important_features = forest_importances.where(forest_importances <= 0.002)
  non_important_features = non_important_features.dropna()

  #important features
  forest_importances = forest_importances.where(forest_importances > 0.002)
  forest_importances = forest_importances.dropna()
  print("Number of features after removing the ones with equal or less than 0.002 importance:", len(forest_importances), "\n")

  #visualization of the top 20 features sorted by importance
  fig, ax = plt.subplots()
  forest_importances.head(20).plot.bar(ax=ax)
  plt.title("Feature importances using MDI")
  ax.set_ylabel("Mean decrease in impurity")
  plt.show()

  #removing non important features from the dataset
  dataset[i].drop(non_important_features.index, axis = 1, inplace=True)


path = os.getcwd() + "/datasets/preprocessed datasets"

#saving the preprocessed datasets
for df in cols:
  with open(path + "/preprocessed_" + df + ".csv", 'w', encoding = 'utf-8-sig') as f:
    dataset[df].to_csv(f)