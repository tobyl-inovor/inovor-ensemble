'''
Ensemble training script for binary classification datasets. 
'''
import numpy as np
import pandas as pd
import math
import os
import sys
import glob
import argparse
import copy

from PIL import Image
import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split

from utils import *
from dl import *
from Clf import *

# configurations
NUM_WORKERS = 8
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 123

# set random seeds for reproducability
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def ensembler(dir, csv, model_type, clf_in, clf_out, loss_fn, resume, ft):
    '''
    Function that performs the ensembling of classification layers using a single feature extractor.
    See main function for input descriptions. 
    '''

    ############## device and model setup ##############
    no_of_classes = 2

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.empty_cache()

    # load fine-tuned model
    if clf_in is not None:
        conv_net, input_size = load_conv_net(model_type, torch.device('cpu'))
        model = Clf()
        cnn = nn.Sequential(
            conv_net,
            model
        )
        checkpoint = torch.load(clf_in)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        conv_net = copy.deepcopy(cnn[0]).to(device)
        model = copy.deepcopy(cnn[1]).to(device)
        del cnn

    else:
        conv_net, input_size = load_conv_net(model_type, device)
        model = Clf().to(device)

    # freeze feature extractor
    for param in conv_net.parameters():
        param.requires_grad = False

    # setup optimizer and learning rate scheduler
    scheduler_step = 15
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.2)

    # print model summary
    summary(conv_net, input_size=(3,224,224))


    ############## augmentations and dataloaders ##############
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

    # load in csv from correct iteration
    if resume is not None:
        resume_iter = int(resume) - 1
        folder_path = clf_out + str(resume_iter) + "/"
        df_train = pd.read_csv(folder_path + "train_set_unique_labels.csv")
        df_val = pd.read_csv(folder_path + "val_set_unique_labels.csv")
        skip_training_once = True
    # case for start - load in original csv and clean data with preprocessing utilities
    else:
        resume_iter = 1
        df = pd.read_csv(csv)
        masks, df, corrupt = classif_imgs_from_masks(df, dir)
        # df = df[0:5000] # line for testing functionality
        df_tv, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
        df_train, df_val = train_test_split(df_tv, test_size=0.20, random_state=RANDOM_SEED)
        skip_training_once = False

    # create datasets
    train_set = BinaryDataset(dir, df_train, transform=transform)
    val_set = BinaryDataset(dir, df_val, transform=test_transform)
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)

    data_loader = {'Training': train_loader, 'Validation': val_loader}
    data_length = {'Training': len(train_loader), 'Validation': len(val_loader)} 


    ############## training settings ##############
    ensemble_iterations = 20
    
    # selection of loss fn for each classifier
    # case for weighted cross entropy each iteration
    if loss_fn is None:
        loss_fn = []
        for i in range(ensemble_iterations):
            loss_fn.append(None)
    
    # case for weighted cross entropy first, then F2 soft loss all following iteratoins
    elif loss_fn.upper() == "FBETA":
        loss_fn = [None]
        for i in range(ensemble_iterations - 1):
            loss_fn.append("fbeta2")
    
    # case for weighted sigmoid or focal loss each iteration
    else:
        loss_fn = [loss_fn]
        loss_fn = loss_fn * ensemble_iterations
    

    ############## ensemble training loop ##############
    for i in range(resume_iter, ensemble_iterations + 1):

        # display class imbalance to user
        print("\nIteration {} has {} samples.".format(i, train_set.labels.shape[0]))
        pos_count = np.sum(train_set.labels)
        samples_per_cls = np.array([train_set.labels.shape[0] - pos_count, pos_count]).flatten()
        total_count = np.sum(samples_per_cls)
        weights_per_cls = torch.tensor(total_count / samples_per_cls).float().to(device)
        print("Samples per class: [{:.0f}, {:.0f}]".format(samples_per_cls[0], samples_per_cls[1]))

        # finish script if all samples belong to same class
        if samples_per_cls[0] == 0 or samples_per_cls[1] == 0:
            print("Ensemble training is finished after iteration {:.0f}.".format(i - 1))
            quit()

        # fine-tune option for first iteration
        if ft.upper() == "TRUE":
            
            # only fine tune once
            ft = "False"
            
            # combine feature extractor and classification layer
            cnn = nn.Sequential(
                conv_net,
                model
            ).to(device)

            # fine tune
            fine_tune(train_set, 
                      val_set, 
                      data_loader, 
                      data_length, 
                      BATCH_SIZE, 
                      cnn, 
                      device, 
                      LEARNING_RATE, 
                      clf_out, 
                      loss_fn[i], 
                      samples_per_cls, 
                      no_of_classes, 
                      weights_per_cls)
            
            # once fine-tuned, load model that gives best validation f1
            folder_path = clf_out + "ft/"
            clf_list = glob.glob( folder_path + "-" + "v-f1" + "*" )
            if len(clf_list) == 1:
                checkpoint = torch.load(clf_list[0])
                cnn.load_state_dict(checkpoint['model_state_dict'])
                print("\n\nLoaded fine tuned cnn...{}\n\n".format(clf_list[0]))
                del conv_net, model
                conv_net = copy.deepcopy(cnn[0]).to(device)
                model = copy.deepcopy(cnn[1]).to(device)
                del cnn

            # skip first iteration of training because we have already fine-tuned
            skip_training_once = True

        # perform training
        if skip_training_once:
            skip_training_once = False
        else:
            clf_trainer(train_set,
                        val_set,
                        data_loader,
                        data_length,
                        NUM_EPOCHS,
                        BATCH_SIZE,
                        conv_net,
                        model,
                        device,
                        optimizer,
                        scheduler,
                        clf_out,
                        loss_fn[i],
                        samples_per_cls,
                        no_of_classes,
                        weights_per_cls,
                        i)

            # load the saved clf with best training loss
            folder_path = clf_out + "/" + str(i) + "/"
            clf_list = glob.glob( folder_path + "-" + "t-loss" + "*" )
            
            if len(clf_list) == 1:
                checkpoint = torch.load(clf_list[0])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                print("\n\nLoaded...{}\n\n".format(clf_list[0]))

        # perform clf on training set (without the augmentations)
        train_set = BinaryDataset(dir, df_train, transform=test_transform)
        pos = clf_tester(train_set,
                         torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
                         BATCH_SIZE,
                         conv_net,
                         model,
                         device)
        
        # create dataset from falsely classified samples
        subset_ind = [j for j, x in enumerate(pos) if not x]
        trues_ind = [j for j, x in enumerate(pos) if x]

        # add some correctly classified samples to dataset with the outliers - 67% outliers, 33% correctly classified samples
        np.random.shuffle(trues_ind)
        try:
            subset_ind.extend(trues_ind[0:round(len(subset_ind)/2)])
        except:
            print("Couldn't add correctly classified samples to dataset with outliers...")

        # setup dataloaders again
        df_train = df_train.iloc[subset_ind]
        train_set = BinaryDataset(dir, df_train, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=NUM_WORKERS)
        data_loader = {'Training': train_loader, 'Validation': val_loader}
        data_length = {'Training': len(train_loader), 'Validation': len(val_loader)}

        # define new classification layer
        model = Clf().to(device)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.2)

        # go to next ensemble iteration ^^^
    
    quit()


############## MAIN ##############
def main():
    parser = argparse.ArgumentParser(description='Run ensemble classifier training.')
    parser.add_argument('--dir',
                       default=None,
                       metavar="/path/to/folder-with-images",
                       help="Full path to the image directory")
    parser.add_argument('--csv',
                       default=None,
                       metavar="/path/to/csv",
                       help="Full path to the csv file containing image names and pixel encodings")
    parser.add_argument('--model_type',
                       default=None,
                       help="Type of model for training")
    parser.add_argument('--clf_in',
                        default=None,
                        metavar='/path/to/input_model',
                        help="Path to classifier, used for loading in fine-tuned model.")
    parser.add_argument('--clf_out',
                        default=None,
                        metavar='/path/to/save_model',
                        help="Path for saving classifier")
    parser.add_argument('--loss_fn',
                        default=None,
                        help="Loss function (sigmoid/focal/fbeta), None gives CrossEntropy")
    parser.add_argument('--resume',
                        default=None,
                        help="Training iteration to resume from (eg. --resume 4 resumes to train 4th classifier")
    parser.add_argument('--ft',
                        default="False",
                        help="Option for fine tuning first iteration of training. Should be used WIHTOUT resume argument.")    

    args = parser.parse_args()
    ensembler(args.dir, args.csv, args.model_type, args.clf_in, args.clf_out, args.loss_fn, args.resume, args.ft)

if __name__ == '__main__':
    main()