#!/usr/bin/env python

from datasets import load_dataset
import pandas as pd
from fmsdemo import CHAT_HISTORY_PATH

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
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["train"])
selected_cols = df[['question', 'answers']]
selected_cols.to_csv(CHAT_HISTORY_PATH)