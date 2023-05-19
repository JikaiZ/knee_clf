import pandas as pd
import numpy as np
import os
import datetime
import argparse
import logging
import json
import random
# Torch utils
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
# sklearn packages
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# scripts
from nlp_models.biobert_model import BERT_classifier
from nlp_models.data_utils_nlp import get_dataloader
from nlp_models.text_preprocess import narrative_cleaner

logger = logging.getLogger(__name__)

def set_seed(args):
	SEED = args.seed

	torch.cuda.empty_cache()
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_text_lengths(input_texts):
	lengths = []
	for text in input_texts:
		lengths.append(len(text))
	logger.info(f'Average length: {int(np.sum(lengths)/len(input_texts))}, SD: {np.std(lengths)}, Max: {np.max(lengths)}, Min: {np.min(lengths)}')

def getTriLabelFreq(input_df):
	input_labels = input_df['TriLabels'].values
	total_num0 = sum(input_labels==0)
	total_num1 = sum(input_labels==1)
	total_num2 = sum(input_labels==2)
	total_num = len(input_df)
	logger.info(f'Total number of Normals: {total_num0}, perc: {total_num0/total_num:.2f}')
	logger.info(f'Total number of Abnormals: {total_num1}, perc: {total_num1/total_num:.2f}')
	logger.info(f'Total number of Excludes: {total_num2}, perc: {total_num2/total_num:.2f}')

def model_preparation(args):
	if args.method == 'BioBERT':
		# model = BertModel.from_pretrained('dmis-lab/biobert-v1.1', return_dict=False)
		model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=3)
		TOKENIZER = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

	return model, TOKENIZER

def plot_loss(train_losses, val_losses, fname, args):
	font = {'family': 'serif',
		'color':  'darkred',
		'weight': 'normal',
		'size': 16,
		}

	x = np.arange(0, len(train_losses), 1, dtype=int)
	plt.figure(figsize=(10, 6))
	plt.plot(x, train_losses, label='Training Loss')
	plt.plot(x, val_losses, label='Val Loss')
	plt.xlabel('Validation steps', fontdict=font)
	plt.ylabel('CE Loss', fontdict=font)
	plt.legend()
	plt.savefig(os.path.join(args.log_dir, fname+"_loss.png"))


def curate_dataloader(args, TOKENIZER):
	PARA_DICT_DIR = os.path.join(args.config_dir, 'nlp/'+args.config_dict+'.txt')
	with open(PARA_DICT_DIR) as f:
		PARA_DICT = f.read()
	PARA_DICT = json.loads(PARA_DICT)
	MAX_SEQ_LEN = PARA_DICT['MAX_SEQ_LEN']
	BATCH_SIZE = PARA_DICT['BATCH_SIZE']

	if args.mode == 'train':
		DATA_PATH = args.data_dir
		DATA_DICT_DIR = os.path.join(args.config_dir, 'data/'+args.data_dict+'.txt')
		with open(DATA_DICT_DIR) as f:
			DATA_DICT = f.read()
		DATA_DICT = json.loads(DATA_DICT)

		train_df = pd.read_excel(os.path.join(DATA_PATH, DATA_DICT['TRAIN']+'.xlsx'))
		val_df = pd.read_excel(os.path.join(DATA_PATH, DATA_DICT['VALIDATION_EVAL']+'.xlsx'))

		logger.info('training file: %s ' % os.path.join(args.data_dir, DATA_DICT['TRAIN']+'.xlsx'))
		logger.info('validation file: %s ' % os.path.join(args.data_dir, DATA_DICT['VALIDATION_EVAL']+'.xlsx'))

		train_label = train_df['TriLabels'].values
		getTriLabelFreq(train_df)
		train_text = narrative_cleaner(train_df)
		get_text_lengths(train_text)

		val_label = val_df['TriLabels'].values
		getTriLabelFreq(val_df)
		val_text = narrative_cleaner(val_df)
		get_text_lengths(val_text)

		_train_seq, _train_mask, train_dataloader = get_dataloader(train_text, train_label, MAX_SEQ_LEN, BATCH_SIZE, TOKENIZER, training=True)
		_val_seq, _val_mask, val_dataloader = get_dataloader(val_text, val_label, MAX_SEQ_LEN, BATCH_SIZE, TOKENIZER, training=False)

		return train_dataloader, val_dataloader, val_df

	if args.mode == 'test':
		test_df = pd.read_excel(os.path.join(args.data_dir, args.test_file+'.xlsx'))
		logger.info(f'Testing file: {os.path.join(args.data_dir, args.test_file)}\n')
		test_label = test_df['TriLabels'].values
		getTriLabelFreq(test_df)
		test_text = narrative_cleaner(test_df)
		get_text_lengths(test_text)
		_test_seq, _test_mask, test_dataloader = get_dataloader(test_text, test_label, MAX_SEQ_LEN, BATCH_SIZE, TOKENIZER, training=False)
		return test_dataloader, test_df

def evaluate(model, val_dataloader, criterion, device):
	# print("\nEvaluating...")
	model.eval()
	total_loss = 0
	total_preds, true_labels, total_probs = [], [], []

	for _, batch in enumerate(val_dataloader):
		batch = [t.to(device) for t in batch]
		sent_id, mask, labels = batch
		true_labels.append(labels.detach().cpu().numpy())

		with torch.no_grad():
			outputs = model(sent_id, attention_mask=mask, labels=labels)
			preds = torch.argmax(outputs.logits, 1)
			probs = torch.nn.functional.softmax(outputs.logits, dim=1)
			# preds = torch.sigmoid(output).detach().cpu().numpy()
			total_preds.append(preds.detach().cpu().numpy())
			total_probs.append(probs.detach().cpu().numpy())
			# loss = criterion(torch.sigmoid(output), labels.type(torch.float))
			# loss = criterion(output, labels)
			loss = outputs.loss
			total_loss = total_loss + loss.item()

	avg_loss = total_loss / len(val_dataloader) 
	total_preds  = np.concatenate(total_preds, axis=0)
	true_labels = np.concatenate(true_labels, axis=0)
	total_probs = np.concatenate(total_probs, axis=0)
	epoch_f1 = f1_score(true_labels, total_preds, average='weighted')
	return avg_loss, total_preds, total_probs, true_labels, epoch_f1

def train(args):
	# set device
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# set seed
	set_seed(args)
	
	# bert
	classifier, TOKENIZER = model_preparation(args)
	classifier = classifier.to(device)

	# data loaders
	train_dataloader, val_dataloader, val_df = curate_dataloader(args, TOKENIZER)
	
	# load h-parameters
	PARA_DICT_DIR = os.path.join(args.config_dir, 'nlp/'+args.config_dict+'.txt')
	with open(PARA_DICT_DIR) as f:
		PARA_DICT = f.read()
	PARA_DICT = json.loads(PARA_DICT)

	PATIENCE = args.patience
	N_EPOCHS = PARA_DICT['EPOCH']
	MAX_SEQ_LEN = PARA_DICT['MAX_SEQ_LEN']
	BATCH_SIZE = PARA_DICT['BATCH_SIZE']
	LEARNING_RATE = PARA_DICT['LEARNING_RATE']
	H = PARA_DICT['H']

	# name of output files
	OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
	# model name
	MODEL_NAME = '{}.pth'.format(OUTPUT_NAME)
	CHECKPOINTS_PATH = os.path.join(args.checkpoints_dir, MODEL_NAME)
	# log file path
	LOG_PATH = os.path.join(args.log_dir, '{}.log'.format(OUTPUT_NAME))
	# results file path
	RESULTS_PATH = os.path.join(args.result_dir, '{}.xlsx'.format(OUTPUT_NAME))

	

	logger.info("TRAINING START")

	optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
	criterion = nn.CrossEntropyLoss()
	# set initial loss to infinite
	min_loss = float('inf')
	best_fscore = 0
	train_losses = []
	val_losses = []

	#for each epoch
	patience_counter = 0
	num_batch = len(train_dataloader)
	midepoch_step = int(num_batch / 2)
	early_stopping_flag = False
	print(f'Number of steps: {num_batch} half-way: {midepoch_step}')
	logger.info(f'Number of steps: {num_batch} half-way: {midepoch_step}')

	for epoch in range(N_EPOCHS):
		print('\nEpoch {:} / {:}'.format(epoch + 1, N_EPOCHS))
		logger.info('\nEpoch {:} / {:}'.format(epoch + 1, N_EPOCHS))

		classifier.train()
		running_loss = step_totals = 0
		total_preds, true_labels, total_probs = [], [], []
		
		for step, batch in enumerate(train_dataloader):
			batch = [r.to(device) for r in batch]
			sent_id, mask, labels = batch

			classifier.zero_grad()        
			outputs = classifier(sent_id, attention_mask=mask, labels=labels)
			preds = torch.argmax(outputs.logits, 1)
			probs = torch.nn.functional.softmax(outputs.logits, dim=1)
			# loss = criterion(output, labels)
			loss = outputs.loss
			running_loss += loss.item() * len(labels)
			step_totals +=  len(labels)
			loss.backward()
			optimizer.step()
			total_probs.append(probs.detach().cpu().numpy())
			total_preds.append(preds.detach().cpu().numpy())
			true_labels.append(labels.detach().cpu().numpy())

			# validate twice in one epoch, at 50% or 100%
			if (step + 1 == midepoch_step) or (step + 1 == num_batch):
				train_loss = running_loss / step_totals
				train_total_preds  = np.concatenate(total_preds, axis=0)
				train_true_labels = np.concatenate(true_labels, axis=0)
				train_total_probs = np.concatenate(total_probs, axis=0)
				train_f1 = f1_score(train_true_labels, train_total_preds, average='weighted')
				val_loss, val_preds, val_probs, val_labels, val_f1 = evaluate(classifier, val_dataloader, criterion, device)
				print(f'\n {(step+1)/num_batch:.0%} Epoch: {epoch+1}')
				print(f' Training Loss: {train_loss}, weighted F1: {train_f1}')
				print(f' Val Loss: {val_loss}, weighted f1: {val_f1}')
				logger.info(f'\n {(step+1)/num_batch:.0%} Epoch: {epoch+1}')
				logger.info(f' Training Loss: {train_loss}, weighted F1: {train_f1}')
				logger.info(f' Val Loss: {val_loss}, weighted f1: {val_f1}')

				train_losses.append(train_loss)
				val_losses.append(val_loss)

				if val_loss < min_loss:
					print(f' Step val loss: {val_loss} BEATS best val loss: {min_loss}\n')
					logger.info(f' Step val loss: {val_loss} BEATS best val loss: {min_loss}\n')
					min_loss = val_loss
					torch.save(classifier.state_dict(), CHECKPOINTS_PATH)
					best_epoch = epoch + 1
					best_probs = val_probs
					best_preds = val_preds
					best_step = step + 1
					best_fscore = val_f1
					patience_counter = 0
				else:
					print(f' Step val loss: {val_loss} DOES NOT BEAT best val loss: {min_loss} for {patience_counter+1} times\n')
					logger.info(f' Step val loss: {val_loss} DOES NOT BEAT best val loss: {min_loss} for {patience_counter+1} times\n')
					patience_counter += 1

			if patience_counter == PATIENCE:
				print(f'Early stopping at {best_step/num_batch:.0%} EPOCH {epoch}')
				logger.info(f'Early stopping at {best_step/num_batch:.0%} EPOCH {epoch}')
				early_stopping_flag = True
				break

		if early_stopping_flag == True:
			break

	print('\nTraining Done!')
	print(f'\nLearning Rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}, Seq Length: {MAX_SEQ_LEN}')
	logger.info(f'\nLearning Rate: {LEARNING_RATE}, Batch size: {BATCH_SIZE}, Seq Length: {MAX_SEQ_LEN}')
	# logger.info(f'Hidden Layers: {FC_DIMS} \n')
	
	logger.info(f'best weighted f1-score with min loss: {best_fscore} at Step: {best_step} EPOCH: {best_epoch} Iteration: {best_epoch*num_batch+best_step}')
	print(f'best weighted f1-score with min loss: {best_fscore} at Step: {best_step} EPOCH: {best_epoch} Iteration: {best_epoch*num_batch+best_step}')

	plot_loss(train_losses, val_losses, OUTPUT_NAME, args)


	print('Validation Starts...')


def test(args):
	# gpu device
	DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# DEVICE = torch.device("cpu")
	# bert
	bert, TOKENIZER = model_preparation(args)
	# data loaders
	test_loader, test_df = curate_dataloader(args, TOKENIZER)	
	# h-parameters
	PARA_DICT_DIR = os.path.join(args.config_dir, 'nlp/'+args.config_dict+'.txt')
	with open(PARA_DICT_DIR) as f:
		PARA_DICT = f.read()
	PARA_DICT = json.loads(PARA_DICT)
	H = PARA_DICT['H']
	if H == 0:
		FC_DIMS = [768, 3]
	else:
		FC_DIMS = [768, H, 3]
	LABELS =  ['Normal', 'Abnormal', 'Exclude']
	N_CLASSES = len(LABELS)

	# name of output files
	OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
	# model name
	MODEL_NAME = '{}.pth'.format(OUTPUT_NAME)
	logger.info(f'Model Name: {MODEL_NAME}\n')
	CHECKPOINTS_PATH = os.path.join(args.checkpoints_dir, MODEL_NAME)
	# results file path
	RESULTS_PATH = os.path.join(args.result_dir, '{}-{}.xlsx'.format(OUTPUT_NAME, args.test_file))

	# eval model
	_classifier = BERT_classifier(bert=bert, fc_dims=FC_DIMS).to(DEVICE)
	criterion = nn.CrossEntropyLoss()
	_classifier.load_state_dict(torch.load(CHECKPOINTS_PATH))
	test_loss, test_preds, test_probs, test_labels, test_f1 = evaluate(_classifier, test_loader, criterion, DEVICE)

	# save results
	classification_df = test_df.copy()
	classification_df[[x + 'p' for x in LABELS]] = test_probs
	classification_df['PredTriLabels'] = test_preds
	classification_df.to_excel(RESULTS_PATH, index=False)

	logger.info(f'weighted f1-score: {test_f1}')


def main():
	parser = argparse.ArgumentParser()
	# paras for generating config files
	parser.add_argument("--task", choices=['MULTICLASS', 'MULTILABEL'], type=str, default='MULTICLASS', help="training task")
	parser.add_argument("--log_dir", default="./log/nlp/", type=str, help="directory of training log")
	parser.add_argument("--result_dir", default="./results/nlp/", type=str, help="directory of saved results")
	parser.add_argument("--data_dir", default="./data/", type=str, help="directory of spreadsheet data for training/validation")
	parser.add_argument("--config_dir", default="./configs/", type=str, help="directory of config files")
	parser.add_argument("--checkpoints_dir", default="./checkpoints/nlp/", type=str, help="directory of saving the best model")

	parser.add_argument("--data_dict", required=True, help="config of training/validation files")
	parser.add_argument("--config_dict", required=True, help="config of hyperparameters")
	parser.add_argument("--test_file", default='VAL_PTEST', type=str, help="test file")
	parser.add_argument("--method", choices=['RB', 'ML', 'BioBERT', 'BERT'], type=str, default='BioBERT', help="methods of training nlp")

	parser.add_argument("--seed", default=123, choices=[1, 42, 100, 111, 123], type=int, help="random seed")
	parser.add_argument("--patience", default=15, type=int, help="patience for early stopping")
	parser.add_argument("--mode", default='train', type=str, choices=['train', 'test'], help="option for testing")

	args = parser.parse_args()

	# set up loggers
	print('Data: %s' % args.data_dict)
	print('Hyperparameters: %s' % args.config_dict)

	if args.mode == 'train':
		OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
		logging.basicConfig(filename=os.path.join(args.log_dir, '{}.log'.format(OUTPUT_NAME)),
				filemode='a',
				format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
				datefmt='%m/%d/%Y %H:%M:%S',
				level=logging.INFO
			)
		train(args)
	if args.mode == 'test':
		OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
		logging.basicConfig(filename=os.path.join(args.log_dir, '{}-{}.log'.format(OUTPUT_NAME, args.test_file)),
			filemode='a',
			format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
			datefmt='%m/%d/%Y %H:%M:%S',
			level=logging.INFO
		)
		test(args)


if __name__ == '__main__':
	main()

