import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import random
import time
from datetime import timedelta
import json
import argparse
import logging

from torchvision import transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from image_models.dataset import KneeDatasetMultiClass

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

def model_preparation(args):
    if args.method == 'effb3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)
        optim = 'Adam'
    if args.method == 'effb4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=3)
        optim = 'Adam'
    if args.method == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 3, bias=False)
        optim = 'SGD'

    return model, optim

def getTriLabelFreq(input_df):
    input_labels = input_df['TriLabels'].values
    total_num0 = sum(input_labels==0)
    total_num1 = sum(input_labels==1)
    total_num2 = sum(input_labels==2)
    total_num = len(input_df)
    logger.info(f'Total number of Normals: {total_num0}, perc: {total_num0/total_num:.2f}')
    logger.info(f'Total number of Abnormals: {total_num1}, perc: {total_num1/total_num:.2f}')
    logger.info(f'Total number of Excludes: {total_num2}, perc: {total_num2/total_num:.2f}')

def data_preparation(args):
    # load data, para files
    with open(os.path.join(args.config_dir, 'image/'+args.config_dict+'.txt')) as f:
        PARA_DICT = f.read()
    PARA_DICT = json.loads(PARA_DICT)

    # data transformations
    simple_trans = transforms.Compose([
        transforms.Resize((PARA_DICT['IMAGE_SIZE'], PARA_DICT['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std  = [ 0.229, 0.224, 0.225 ]),
        ])

    if args.mode == 'train':

        with open(os.path.join(args.config_dir, 'data/'+args.data_dict+'.txt')) as f:
            DATA_DICT = f.read()
        DATA_DICT = json.loads(DATA_DICT)

        TRAIN_PRI_df = pd.read_excel(os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'))
        val_df = pd.read_excel(os.path.join(args.data_dir, DATA_DICT['VALIDATION_EVAL']+'.xlsx'))

        # determine training files
        if DATA_DICT['PLABEL_PERC'] == 'customize':
            P_LABEL_df = pd.read_excel(os.path.join(args.autolabel_dir, args.plabel_file)+'.xlsx')
            train_df = pd.concat([TRAIN_PRI_df, P_LABEL_df]).reset_index(drop=True)
            logger.info('training file: %s + %s' % (os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'), args.plabel_file))
        else:
            if DATA_DICT['PLABEL_PERC'] == 0:
                train_df = TRAIN_PRI_df
                print('training file: %s ' % os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'))
                logger.info('training file: %s ' % os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'))
            elif DATA_DICT['PLABEL_PERC'] == 1:
                P_LABEL_df = pd.read_excel(os.path.join(args.autolabel_dir, '{}.xlsx'.format(args.plabel_file)))
                train_df = pd.concat([TRAIN_PRI_df, P_LABEL_df]).reset_index(drop=True)
                print('training file: %s + %s' % (os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'), '{}.xlsx'.format(args.plabel_file)))
                logger.info('training file: %s + %s' % (os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'), '{}.xlsx'.format(args.plabel_file)))
            elif (DATA_DICT['PLABEL_PERC'] > 0) and (DATA_DICT['PLABEL_PERC'] < 1):
                P_LABEL_df = pd.read_excel(os.path.join(args.autolabel_dir, '{}{}p.xlsx'.format(args.plabel_file, str(int(DATA_DICT['PLABEL_PERC']*100)))))
                train_df = pd.concat([TRAIN_PRI_df, P_LABEL_df]).reset_index(drop=True)
                logger.info('training file: %s + %s' % (os.path.join(args.data_dir, DATA_DICT['TRAIN_PRI']+'.xlsx'), 
                    '{}{}p.xlsx'.format(args.plabel_file, str(int(DATA_DICT['PLABEL_PERC']*100)))))
            else:
                print("Error, wrong perc input.")

        print('validation file: %s ' % os.path.join(args.data_dir, DATA_DICT['VALIDATION_EVAL']+'.xlsx'))
        logger.info('validation file: %s ' % os.path.join(args.data_dir, DATA_DICT['VALIDATION_EVAL']+'.xlsx'))

        # data loaders
        train_dataset = KneeDatasetMultiClass(input_df = train_df,
            root_dir = args.image_dir,
            transform = simple_trans)
        train_loader = DataLoader(dataset=train_dataset, 
            batch_size=PARA_DICT['BATCH_SIZE'], 
            shuffle=True)
        valid_dataset = KneeDatasetMultiClass(input_df = val_df,
            root_dir = args.image_dir,
            transform = simple_trans)
        valid_loader = DataLoader(dataset=valid_dataset, 
            batch_size=PARA_DICT['BATCH_SIZE'], 
            shuffle=False)

        return train_loader, valid_loader, train_df, val_df

    if args.mode == 'test':
        test_df = pd.read_excel(os.path.join(args.data_dir, args.test_file+'.xlsx'))
        logger.info(f'Testing file: {os.path.join(args.data_dir, args.test_file)}\n')
        test_dataset = KneeDatasetMultiClass(input_df = test_df,
            root_dir = args.image_dir,
            transform = simple_trans)
        test_loader = DataLoader(dataset=test_dataset, 
            batch_size=PARA_DICT['BATCH_SIZE'], 
            shuffle=False)
        return test_loader, test_df


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

def evaluate(model, criterion, data_loader, N_CLASSES, DEVICE):
    total_preds, true_labels, total_probs = [], [], []
    step_totals = 0

    running_loss = 0
    with torch.no_grad():
        model.eval()
        for idx, (data, label) in enumerate(data_loader):
            data = data.to(DEVICE)

            label = label.to(DEVICE)

            logits = model(data)
            loss = criterion(logits, label)
            step_totals +=  len(label)

            running_loss += loss.item() * len(label)

            total_probs.append(torch.nn.functional.softmax(logits,1).detach().cpu().numpy())
            total_preds.append(torch.argmax(logits, 1).detach().cpu().numpy())
            true_labels.append(label.detach().cpu().numpy())


    total_preds  = np.concatenate(total_preds, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    total_probs = np.concatenate(total_probs, axis=0)

    epoch_loss = running_loss / step_totals

    epoch_preds = total_preds
    epoch_probs = total_probs
    epoch_f1 = f1_score(true_labels, epoch_preds, average='weighted')

    return epoch_loss, epoch_probs, epoch_preds, epoch_f1

def train(args):
    set_seed(args)
    start = time.time()
    # gpu device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # data loaders
    train_loader, valid_loader, train_df, val_df = data_preparation(args)
    train_labels, val_labels = train_df['TriLabels'].values, val_df['TriLabels'].values

    logger.info('Training labels:')
    getTriLabelFreq(train_df)
    logger.info('Validation labels:')
    getTriLabelFreq(val_df)
    
    # h-parameters
    PARA_DICT_DIR = os.path.join(args.config_dir, 'image/'+args.config_dict+'.txt')
    with open(PARA_DICT_DIR) as f:
        PARA_DICT = f.read()
    PARA_DICT = json.loads(PARA_DICT)

    N_EPOCHS = PARA_DICT['EPOCH']
    LEARNING_RATE = PARA_DICT['LEARNING_RATE']
    N_CLASSES = 3
    PATIENCE = args.patience
    # name of output files
    OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
    # model name
    MODEL_NAME = '{}.pth'.format(OUTPUT_NAME)
    CHECKPOINTS_PATH = os.path.join(args.checkpoints_dir, MODEL_NAME)
    # log file path
    LOG_PATH = os.path.join(args.log_dir, '{}.log'.format(OUTPUT_NAME))
    # results file path
    RESULTS_PATH = os.path.join(args.result_dir, '{}.xlsx'.format(OUTPUT_NAME))

    # deep learning model specs
    model, optim = model_preparation(args)
    model = model.to(DEVICE)
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # start training
    min_best_fscore = f1_score(val_labels, [1] * len(val_labels), average='weighted')
    best_fscore = 0
    min_loss = 1e10
    best_epoch = 0
    best_probs = []
    best_step = 0
    iter_num = 0
    train_losses = []
    val_losses = []
    print('start training...')
    logger.info('start training...')
    patience_counter = 0
    num_batch = len(train_loader)
    midepoch_step = int(num_batch / 2)
    early_stopping_flag = False
    print(f'Number of steps: {num_batch} half-way: {midepoch_step}')
    logger.info(f'Number of steps: {num_batch} half-way: {midepoch_step}')
    # N_EPOCHS = 1
    for epoch in range(N_EPOCHS):
        logger.info('\nEpoch {:} / {:}'.format(epoch + 1, N_EPOCHS))
        print('\nEpoch {:} / {:}'.format(epoch + 1, N_EPOCHS))

        running_loss = step_totals = 0
        total_preds, true_labels, total_probs = [], [], []
        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        
            running_loss += loss.item() * len(label)
            step_totals +=  len(label)
            
            total_probs.append(torch.nn.functional.softmax(logits,1).detach().cpu().numpy())
            total_preds.append(torch.argmax(logits, 1).detach().cpu().numpy())
            true_labels.append(label.detach().cpu().numpy())

            # validate twice in one epoch, at 50% or 100%
            if (step + 1 == midepoch_step) or (step + 1 == num_batch):
                train_loss = running_loss / step_totals
                train_total_preds  = np.concatenate(total_preds, axis=0)
                train_true_labels = np.concatenate(true_labels, axis=0)
                train_total_probs = np.concatenate(total_probs, axis=0)
                train_f1 = f1_score(train_true_labels, train_total_preds, average='weighted')
                val_loss, val_probs, val_preds, val_f1 = evaluate(model, criterion, valid_loader, N_CLASSES, DEVICE)
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
                    torch.save(model.state_dict(), CHECKPOINTS_PATH)
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

    logger.info(f'best weighted f1-score with min loss: {best_fscore} at Step: {best_step} EPOCH: {best_epoch} Iteration: {best_epoch*num_batch+best_step}')
    print(f'best weighted f1-score with min loss: {best_fscore} at Step: {best_step} EPOCH: {best_epoch} Iteration: {best_epoch*num_batch+best_step}')

    elapsed = time.time() - start
    print(f'Training done! Elapsed time: {str(timedelta(seconds = elapsed))}')
    logger.info(f'Training done! Elapsed time: {str(timedelta(seconds = elapsed))}')
    print(PARA_DICT)
    logger.info(PARA_DICT)

    plot_loss(train_losses, val_losses, OUTPUT_NAME, args)


def test(args):
    # gpu device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    # data loaders
    test_loader, test_df = data_preparation(args)
    test_labels = test_df['TriLabels'].values

    logger.info("Test labels:")
    getTriLabelFreq(test_df)
    
    # h-parameters
    PARA_DICT_DIR = os.path.join(args.config_dir, 'image/'+args.config_dict+'.txt')
    with open(PARA_DICT_DIR) as f:
        PARA_DICT = f.read()
    PARA_DICT = json.loads(PARA_DICT)
    LABELS =  ['Normal', 'Abnormal', 'Exclude']
    N_CLASSES = len(LABELS)

    # name of output files
    OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
    # model name
    MODEL_NAME = '{}.pth'.format(OUTPUT_NAME)
    logger.info(f'Model Name: {MODEL_NAME}\n')
    CHECKPOINTS_PATH = os.path.join(args.checkpoints_dir, MODEL_NAME)
    # results file path
    RESULTS_PATH = os.path.join(args.result_dir, '{}-{}.xlsx'.format(OUTPUT_NAME,args.test_file))

    # eval model
    model, _ = model_preparation(args)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(CHECKPOINTS_PATH))
    test_loss, test_probs, test_preds, test_f1 = evaluate(model, criterion, test_loader, N_CLASSES, DEVICE)

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
    parser.add_argument("--log_dir", default="./log/image/", type=str, help="directory of training log")
    parser.add_argument("--result_dir", default="./results/image/", type=str, help="directory of saved results")
    parser.add_argument("--data_dir", default="./data/", type=str, help="directory of spreadsheet data for training/validation")
    parser.add_argument("--config_dir", default="./configs/", type=str, help="directory of config files")
    parser.add_argument("--checkpoints_dir", default="./checkpoints/image/", type=str, help="directory of saving the best model")
    parser.add_argument("--autolabel_dir", default="./autolabels/", type=str, help="directory of autolabels")
    parser.add_argument("--image_dir", default="../../../images_8bit/BLPA2019/", type=str, help="directory of images")

    parser.add_argument("--data_dict", required=True, help="config of training/validation files")
    parser.add_argument("--config_dict", required=True, help="config of hyperparameters")
    parser.add_argument("--test_file", default="VAL_PTEST", type=str, help="test file")
    parser.add_argument("--method", choices=['effb3', 'effb4', 'resnet50', 'inceptionv3'], type=str, default='eff-b4', help="methods of training image")
    parser.add_argument("--plabel_file", type=str, help="pseudo label file")

    parser.add_argument("--seed", default=123, choices=[1, 42, 100, 111, 123], type=int, help="random seed")
    parser.add_argument("--patience", default=10, type=int, help="patience for early stopping")
    parser.add_argument("--mode", default='train', type=str, choices=['train', 'test'], help="option for testing")

    args = parser.parse_args()

    # set up loggers
    ## name of output files
    

    print('Data: %s' % args.data_dict)
    print('Hyperparameters: %s' % args.config_dict)

    # plot_loss([0.8, 0.7, 0.4, 0.2, 0.05], [0.98, 0.57, 0.44, 0.21, 0.015], 'xxx', args)

    if args.mode == 'train':
        OUTPUT_NAME = '{}-{}-SEED{}'.format(args.config_dict, args.data_dict, args.seed)
        logging.basicConfig(filename=os.path.join(args.log_dir, '{}.log'.format(OUTPUT_NAME)),
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        logging.info('\n Training %s starts...\n' % OUTPUT_NAME)
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

