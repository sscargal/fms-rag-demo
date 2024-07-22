#!/usr/bin/env python

import pandas as pd

#load the dataset from hugging face
#dataset = load_dataset("dell-research-harvard/newswire")
#
##combine text data from all splits (train, validate, test) into a single file (may need to change if evaluating output)
#FILE_PATH = "test_data/newswire.txt"
#print("Writing newswire dataset to text file")
#with open(FILE_PATH, "w", encoding="utf-8") as outfile:
#    for split in dataset.keys():
#        for item in dataset[split]:
#            outfile.write(item['text'] + "\n")  
#
#print("Dataset downloaded and combined into 'f{FILE_PATH}'")

#download and parse question-answer pair dataset
CHAT_HISTORY_PATH = "mock_chat_history.csv"
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["train"])
selected_cols = df[['question', 'answers']]
selected_cols.to_csv(CHAT_HISTORY_PATH)

#EN_WIKI_LINES = 5000
#print("Reading csv")
#df = pd.read_json("hf://datasets/vocab-transformers/wiki-en-passages-20210101/train.jsonl.gz", lines=True)
#print("Writing csv lines to file")
#with open("../data/en_wiki.csv", "w") as f:
#    i = 0
#    for row in df.iterrows:
#        if i >= 5000:
#            break
#        f.write(row["text"])
#        i += 1 