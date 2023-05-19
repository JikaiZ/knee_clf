import pandas as pd
import numpy as np
import re
import string

def remove_signatures(input_string):
	tokens = input_string.split(' ')
	signature_idx = []
	for idx, token in enumerate(tokens):
	    if 'electronically' in token.lower():
	        signature_idx.append(idx)
	if len(signature_idx) != 0:
	    first_signature_idx = signature_idx[0]
	    new_tokens = ' '.join(tokens[:first_signature_idx])
	else:
	    new_tokens = ' '.join(tokens)
	return new_tokens


def label_encoder(input_df):
    """encode labels and add the encoded labels to another column"""
    label_dict = {'normal': 0, 'abnormal': 1, 'exclude': 2}
    input_df['label'] = input_df.Label.str.lower().replace(label_dict)
    return input_df.label.values

def clip_reports(input_string):
    tokens = input_string.split(' ')
    start_id, end_id = 0, len(tokens)
    find_start, find_end = False, False
    find_hx_start, find_hx_end = False, False
    hx_start_id, hx_end_id = 0, 0
    # find first locations of findings and impression section title
    # such locations were starting point of the report
    # if neither of those locations exist in the report, find the first location of history
    # then we will try to skip the history section in later segment because history contains DX code info
    # which is not in our scope of annotation
    for idx, token in enumerate(tokens):
        if (('findings' in token.lower()) or ('impression' in token.lower())) and (find_start == False):
            start_id = idx
            find_start = True
        if ('history' in token.lower()) and (find_hx_start == False):
            hx_start_id = idx
            find_hx_start = True
  
    # Skip history if no findings/impression
    # in this case, the we started from the position #0 of the report
    if (find_hx_start == True) and (find_start == False):
        _ = hx_start_id + 1
        # p counter refers to the number of periods . in the report
        
        p_counter = 0
        test_string = ' '.join(tokens[_:])
        
        # In hx, there are TWO periods: one in the DX code, the other at the end of the hx section
        # we skip the hx section by finding exactly TWO periods after the hx section title
        # no dx code for knee replacement, so the maximum number of periods in this special case is ONE
        if 'Knee replacement' in test_string:
            require_p_count = 1
        else:
            require_p_count = 2

        while find_hx_end == False:
            token = tokens[_]
            if '.' in token:
                p_counter += 1
            if p_counter == require_p_count:
                find_hx_end = True
                hx_end_id = _
                start_id = hx_end_id + 1
            _ += 1
    
    # We also remove the electronic signature after the starting position
    # The exlectronic signature starts with the word "electronically"
    # skip all the subsequent words.
    for idx, token in enumerate(tokens[start_id:]):
        if ('electronically' in token.lower()) and (find_end == False):
            end_id = start_id + idx 
            find_end = True

    res = ' '.join(tokens[start_id:end_id])
    return res


def narrative_cleaner(input_df):
    """preprocessing texts"""
    removed_punc = r'!"#$%&\'()*+,-/;<=>?@[\\]^_`{|}~'
    res_df = input_df.copy()
    reports = []
    for idx, row in input_df.iterrows():
        report = row.Narrative
        cleaned_report = clip_reports(report)
        cleaned_report = re.sub('[0-9]+', '', cleaned_report)
        cleaned_report = cleaned_report.translate(str.maketrans('', '', removed_punc))
        reports.append(cleaned_report)
    #     res_df['text'] = reports
    return reports