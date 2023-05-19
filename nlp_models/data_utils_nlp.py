import pandas as pd
import numpy as np
import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from .text_preprocess import narrative_cleaner

def get_dataloader(text, labels, MAX_SEQ_LEN, BATCH_SIZE, tokenizer, training=True):
    tokens = tokenizer.batch_encode_plus(
        text,
        max_length = MAX_SEQ_LEN,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )
    _seq = torch.tensor(tokens['input_ids'])
    _mask = torch.tensor(tokens['attention_mask'])
    _y = torch.tensor(labels.tolist())
    
    _data = TensorDataset(_seq, _mask, _y)
    if training == True:
        # sampler for sampling the data during training
        _sampler = RandomSampler(_data)
    else:
        _sampler = SequentialSampler(_data)
    
    _dataloader = DataLoader(_data, sampler = _sampler, batch_size=BATCH_SIZE)
    return _seq, _mask, _dataloader


def get_text_lengths(input_texts):
	lengths = []
	for text in input_texts:
		lengths.append(len(text))
	print(f'Average length: {int(np.sum(lengths)/len(input_texts))}, SD: {np.std(lengths)}, Max: {np.max(lengths)}, Min: {np.min(lengths)}')

def getTriLabelFreq(input_df):
	input_labels = input_df['TriLabels'].values
	total_num0 = sum(input_labels==0)
	total_num1 = sum(input_labels==1)
	total_num2 = sum(input_labels==2)
	total_num = len(input_df)
	print(f'Total number of Normals: {total_num0}, perc: {total_num0/total_num:.2f}')
	print(f'Total number of Abnormals: {total_num1}, perc: {total_num1/total_num:.2f}')
	print(f'Total number of Excludes: {total_num2}, perc: {total_num2/total_num:.2f}')