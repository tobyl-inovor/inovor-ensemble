import numpy as np
import pandas as pd
import math
import os
import sys
import glob
import argparse
import pickle
import copy
from pprint import pprint
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import imblearn

from PIL import Image
import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# from utils import *
from utils import *
from Clf import *
from dl import *

# configurations
NUM_WORKERS = 8
NUM_CLASSES = 2
BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
RANDOM_SEED = 123

# set random seeds for reproducability
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def meta_trainer(dir, csv, model_type, data_in, data_out, ens_num, ens_type, pos_weight, ft):
    '''
    Function that trains a meta learner - designed to be used after train_clfs.py. 
    
    data_out [str] is not None: function will pass images through feature extractor + ensemble of classifiers
        and save the logit outputs as numpy files.

    data_out [str] is None: function will train a meta-learner on the previously saved numpy files. 
    
    See main function for other input descriptions. 
    '''

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.empty_cache()

    # make ens_num variables integers
    ens_num = int(ens_num)

    # load feature extractor
    conv_net, input_size = load_conv_net(model_type, device)

    for param in conv_net.parameters():
        param.requires_grad = False

    # load in individual classifiers that attained best training loss
    clf_list = []
    for i in range(ens_num):
        clf_list.append(glob.glob( data_in + str(i+1) + "/-" + "t-loss" + "*" ))

    checkpoints = []
    for i in range(len(clf_list)):
        temp = torch.load(clf_list[i][0])
        checkpoints.append(temp)

    clfs = []
    for i in range(len(checkpoints)):
        clfs.append(Clf().to(device))
        clfs[i].load_state_dict(checkpoints[i]['model_state_dict'])

    # option for loading in fine-tuned CNN - loads in feature extraction with best validation f1
    if ft.upper() == "TRUE":
        conv_net, input_size = load_conv_net(model_type, torch.device('cpu'))
        model = Clf()
        cnn = nn.Sequential(
            conv_net,
            model
        )
        temp = glob.glob( data_in + "ft/-v-f1*" )
        checkpoint = torch.load(temp[0])
        cnn.load_state_dict(checkpoint['model_state_dict'])
        conv_net = copy.deepcopy(cnn[0]).to(device)
        clfs[0] = copy.deepcopy(cnn[1]).to(device)
        del cnn
    
    summary(clfs[0], input_size=(1,1280))

    # define augmentation pipeline
    transform = transforms.Compose(
        [transforms.Resize(input_size, Image.BICUBIC),
         transforms.RandomHorizontalFlip(p=0.25),
         transforms.RandomVerticalFlip(p=0.25),
         transforms.RandomRotation(5),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05, saturation=0.05),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose(
        [transforms.Resize(input_size, Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load in pandas df
    df = pd.read_csv(csv)
    masks, df, corrupt = classif_imgs_from_masks(df, dir)
    df_tv, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_train, df_val = train_test_split(df_tv, test_size=0.20, random_state=RANDOM_SEED)

    # case for generating boosting data
    if data_out is not None:

        # create dataset
        train_set = BinaryDataset(dir, df_train, transform=transform)
        val_set = BinaryDataset(dir, df_val, transform=test_transform)
        test_set = BinaryDataset(dir, df_test, transform=test_transform)

        # create dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=NUM_WORKERS)
        data_loader = {'Training': train_loader, 'Validation': val_loader, 'Testing': test_loader}
        data_length = {'Training': len(train_loader), 'Validation': len(val_loader), 'Testing': len(test_loader)}
        
        # save boosting data
        create_meta_data(train_set, val_set, test_set, data_loader, data_length, BATCH_SIZE, conv_net, device, data_in, clfs)
        quit()

    # case for training boosting model (i.e. boosting data has already been generated)
    else:
        if ens_type is None:
            print("\n\nSupply command-line argument 'ens_type', e.g. '--ens_type xgboost'\n\n")
            quit()

        # load ensemble data
        X_train = np.load(data_in + "boost_data_train.npy")
        y_train = np.load(data_in + "boost_labels_train.npy")
        X_val = np.load(data_in + "boost_data_val.npy")
        y_val = np.load(data_in + "boost_labels_val.npy")
        X_test = np.load(data_in + "boost_data_test.npy")
        y_test = np.load(data_in + "boost_labels_test.npy")

        # truncate to number of clfs desired
        X_train = X_train[:,0:ens_num*2]
        X_val = X_val[:,0:ens_num*2]
        X_test = X_test[:,0:ens_num*2]

        # mask NaNs from training data
        for i in range(X_train.shape[0]):
            if np.isnan(X_train[i,:]).any():
                X_train[i,:] = 0
                y_train[i] = 0
    
        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier

        if ens_type.upper() == "ADABOOST":
            meta_clf = AdaBoostClassifier(base_estimator=None, n_estimators=100, random_state=RANDOM_SEED)
        elif ens_type.upper() == "EXTRATREES":
            meta_clf = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, verbose=True, class_weight='balanced')
        elif ens_type.upper() == "XGBOOST":
            meta_clf = XGBClassifier(objective='binary:logistic', max_depth=16, label_encoder=False, subsample=1, n_estimators=300, eta=0.1,
                                     scale_pos_weight=pos_weight)
        elif ens_type.upper() == "RANDOMFOREST":
            meta_clf = RandomForestClassifier(verbose=1, class_weight={0: 1, 1: pos_weight})
        elif ens_type.upper() == "BAGGING":
            meta_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=9))
        elif ens_type.upper() == "LOGISTIC":
            meta_clf = LogisticRegression()
        elif ens_type.upper() == "MLP":
            meta_clf = MLPClassifier(hidden_layer_sizes=ens_num, activation='relu', solver='adam', verbose=True, learning_rate='adaptive',
                                     random_state=RANDOM_SEED)
        elif ens_type.upper() == "KNN":
            meta_clf = KNeighborsClassifier(n_neighbors=5)
        elif ens_type.upper() == "GRADIENTBOOST":
            meta_clf = GradientBoostingClassifier()

        # fit meta learner
        meta_clf.fit(X_train, y_train)

        # print statistics
        print("\nTraining score: ", meta_clf.score(X_train, y_train))
        f1_train = f1_score(y_train, meta_clf.predict(X_train), average=None, labels=[0,1]).tolist()
        print("Training F1: ", f1_train[1])

        print("\nValidation score: ", meta_clf.score(X_val, y_val))
        f1_val = f1_score(y_val, meta_clf.predict(X_val), average=None, labels=[0,1]).tolist()
        print("Validation F1: ", f1_val[1])

        test_preds = meta_clf.predict(X_test)
        print("\n\nTest score: ", meta_clf.score(X_test, y_test))
        f1_test = f1_score(y_test, test_preds, average=None, labels=[0,1]).tolist()
        print("Testing F1: ", f1_test[1])

        # count false positives and false negatives
        fn_and_fp(y_test, test_preds)

        # calculate single classifier statistics:
        single_clf_test_preds = np.argmax(X_test[:,0:2], axis=1)
        print("\nSingle classifier test score: ", accuracy_score(y_test, single_clf_test_preds))
        f1_test_single = f1_score(y_test, single_clf_test_preds, average=None, labels=[0,1]).tolist()
        print("Single classifier test F1: ", f1_test_single[1])
        fn_and_fp(y_test, single_clf_test_preds)

        # calculate model averaging statistics
        test_preds_avg = model_average(softmax_transform(X_test))
        print("\nModel average test score: ", accuracy_score(y_test, test_preds_avg))
        f1_test_avg = f1_score(y_test, test_preds_avg, average=None, labels=[0,1]).tolist()
        print("Model average test F1: ", f1_test_avg[1])
        fn_and_fp(y_test, test_preds_avg)

    quit()


############## MAIN ##############
def main():
    parser = argparse.ArgumentParser(description='Run meta learning after train_clfs.py script.')
    parser.add_argument('--dir',
                       default=None,
                       metavar="/path/to/folder-with-images",
                       help="Full path to the image directory")
    parser.add_argument('--csv',
                       default=None,
                       metavar="/path/to/csv",
                       help="Full path to the csv file containing image names and pixel encodings")
    parser.add_argument('--model_type',
                        required=True,
                        default=None,
                        help="Type of model for training")
    parser.add_argument('--data_in',
                        default=None,
                        metavar='/path/to/input_model',
                        help="Path to directory containing classification models (corresponds to clf_out from train_clfs.py)")
    parser.add_argument('--data_out',
                        default=None,
                        help="Enter any input here if wanting to create and save meta-data.")
    parser.add_argument('--ens_num',
                        required=True,
                        default=10,
                        help="Number of classifiers used for ensemble - important when data_out is not None")
    parser.add_argument('--ens_type',
                        required=False,
                        help="Type of network for meta learner (xgboost/adaboost/logistic/randomforest/extratrees/mlp/bagging)")
    parser.add_argument('--pos_weight',
                        required=False,
                        default=2,
                        help="Weighting for positive (minority) class when using XGBOOST")
    parser.add_argument('--ft',
                        default="False",
                        help="Option for loading in feature extractor that has been fine-tuned.") 

    args = parser.parse_args()

    meta_trainer(args.dir, args.csv, args.model_type, args.data_in, args.data_out, args.ens_num, args.ens_type, args.pos_weight, args.ft)

if __name__ == '__main__':
    main()