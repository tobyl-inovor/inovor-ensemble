import sys
import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.special import softmax
from Clf import *
from CBLoss import *
from FBeta_Loss import *

def load_conv_net(model_type, device):
    '''
    Function to load a CNN based on string input.

    Inputs:
        model_type [str]: specifies type of pre-trained model to load
        device [torch.device]: cpu or gpu 
            instantiated using torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Outputs:
        conv_net [torch.nn]: CNN loaded onto specified device
        input_size: image size suitable for CNN
    '''
    if model_type is not None:
        if model_type.upper() == "EFFICIENTNET-B0":
            from efficientnet_pytorch import EfficientNet
            conv_net = EfficientNet.from_pretrained('efficientnet-b0').to(device)
            conv_net._dropout = Identity()
            conv_net._fc = Identity()
            input_size = 224
        
        elif model_type.upper() == "EFFICIENTNET-B1":
            from efficientnet_pytorch import EfficientNet
            conv_net = EfficientNet.from_pretrained('efficientnet-b1').to(device)
            conv_net._dropout = Identity()
            conv_net._fc = Identity()
            input_size = 240
        
        elif model_type.upper() == "MOBILENET-V2":
            conv_net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
            conv_net.classifier = Identity()
            input_size = 224

        else:
            print("Choose valid model.")
            quit()
        return conv_net, input_size


def choose_loss_fn(loss_fn, samples_per_cls, no_of_classes, weights_per_cls):
    '''
    Function that returns loss function object based on string input. 

    Inputs:
        loss_fn [str]: specifies loss function to use
            None: weighted cross entropy loss
            fbetaX: FX soft loss (if X=1, then F1 soft loss)
            focal: class balanced focal loss
            sigmoid: class balanced sigmoid loss
        samples_per_cls [np.array(n,)]: array storing count of samples for each of the n classes
        no_of_classes [int]: number of classes 
        weights_per_cls [torch.tensor(n,)]: torch tensor of weights for each of the n classes, loaded onto device

    Outputs:
        criterion [torch.nn]: loss function object
    '''
    if loss_fn is None:
        criterion = nn.CrossEntropyLoss(weight=weights_per_cls)
    elif loss_fn[0:5] == "fbeta":
        beta_val = float(loss_fn[5:])
        criterion = FBetaLoss(beta=beta_val)
    elif loss_fn == "focal" or loss_fn == "sigmoid":
        criterion = CBLoss(samples_per_cls, no_of_classes, loss_fn, 0.9999, 0.5)
    return criterion


def save_model(model, optimizer, epoch, accuracy, save_train_loss, save_val_loss, save_train_f1, 
               save_val_f1, save_train_acc, save_val_acc, clf_out, iteration, fine_tune):
    '''
    Function called within training which saves model to specified path. Saves model if model has best training/validation
    loss, F1, or accuracy. Overrides models of previous best performance in each of the categories. 

    Inputs:
        model [torch.nn]: model to be saved
        optimizer [torch.optim]: optimizer object (also saved)
        epoch [int]: epoch number
        accuracy [float]: accuracy of classification for the given epoch
        save_train_loss [bool]: true if saving model as it achieved best training loss
        save_val_loss [bool]: true if saving model as it achieved best validation loss
        save_train_f1 [bool]: true if saving model as it achieved best training F1
        save_val_f1 [bool]: true if saving model as it achieved best validation F1
        save_train_acc [bool]: true if saving model as it achieved best training accuracy
        save_val_acc [bool]: true if saving model as it achieved best validation accuracy
        clf_out [str]: path to where model is saved
        iteration [int]: current ensembling iteration
        fine_tune [bool]: true if saving a model that is being fine-tuned
    '''
    save_cond = [save_train_loss, save_val_loss, save_train_f1, save_val_f1, save_train_acc, save_val_acc]
    prefix = ["t-loss", "v-loss", "t-f1", "v-f1", "t-acc", "v-acc"]

    for i in range(len(save_cond)):

        if clf_out is not None and save_cond[i]:

            # delete previous model with same prefix
            if fine_tune:
                folder_path = clf_out + "ft/"
            else:
                folder_path = clf_out + str(iteration) + "/"
            file_to_del = glob.glob( folder_path + "-" + prefix[i] + "*" )
            if len(file_to_del) == 1:
                os.remove(file_to_del[0])

            # save clf
            clf_save = folder_path + "-{}-{:02d}-{:.3f}-clf.tar".format(prefix[i], epoch, accuracy)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, clf_save)

    return


def print_scores(running_loss, data_length, phase, gt, pred):
    '''
    Utility function for printing model statistics - handled within training / testing / fine-tuning functions
    '''
    epoch_loss = running_loss / data_length[phase]
    accuracy = accuracy_score(gt[phase].detach().cpu(), pred[phase].detach().cpu())
    f1_class_scores = f1_score(gt[phase].detach().cpu(), pred[phase].detach().cpu(), average=None, labels=[0,1]).tolist()
    string_out = "                                                                             "
    sys.stdout.write('%s\r' % string_out)
    sys.stdout.flush()
    if running_loss is not None:
        print('{} \tLoss: {:.4f}, \tAccuracy: {:.4f}, \tF1: [{:.4f}, {:.4f}]'
                .format(phase, epoch_loss, accuracy, f1_class_scores[0], f1_class_scores[1]))
    return epoch_loss, accuracy, f1_class_scores


def clf_trainer(train_set, val_set, data_loader, data_length, end_epoch, batch_size, 
                           conv_net, model, device, optimizer, scheduler, clf_out, loss_fn, 
                           samples_per_cls, no_of_classes, weights_per_cls, iteration):
    '''
    Function for training a classification layer with fixed feature extractor. 
    
    Inputs:
        train_set [BinaryDataset]: Dataset object for training set
        val_set [BinaryDataset]: Dataset object for validation set
        data_loader [dict]: dict of data loader objects 
            e.g. data_loader = {'Training': train_loader, 'Validation': val_loader}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        data_length [dict]: dict of data loader lengths
            e.g. data_length = {'Training': len(train_loader), 'Validation': len(val_loader)}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        end_epoch [int]: number of epochs desired for training
        batch_size [int]: batch size in training
        conv_net [torch.nn]: feature extractor, loaded to device
        model [torch.nn]: classification layer, loaded to device
        device [torch.device]: cpu or gpu
            instantiated using torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer [torch.optim]: optimizer object
        scheduler [torch.optim.lr_scheduler]: learning rate scheduler object
        clf_out [str]: path to where model is saved
        loss_fn [str]: specifies loss function to use
        samples_per_cls [np.array(n,)]: array storing count of samples for each of the n classes
        no_of_classes [int]: number of classes 
        weights_per_cls [torch.tensor(n,)]: torch tensor of weights for each of the n classes, loaded onto device
        iteration [int]: iteration in ensembling (i.e. 3rd iteration trains the 3rd classification layer)
    '''
    # save csv for training and validation sets
    folder_path = clf_out + str(iteration) + "/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    train_set.df.to_csv(folder_path + "train_set_unique_labels.csv")
    val_set.df.to_csv(folder_path + "val_set_unique_labels.csv")

    # define variables for calculating statistics
    pred_train = torch.zeros(len(train_set)).to(device)
    labels_train = torch.zeros(len(train_set)).to(device)
    pred_val = torch.zeros(len(val_set)).to(device)
    labels_val = torch.zeros(len(val_set)).to(device)
    pred = {'Training': pred_train, 'Validation': pred_val}
    gt = {'Training': labels_train, 'Validation': labels_val}
    min_train_loss = 999999999
    min_val_loss = 999999999
    max_train_f1 = 0
    max_val_f1 = 0
    max_train_acc = 0
    max_val_acc = 0

    # put feature extractor in evaluation mode (important for layers such as BN)
    conv_net.eval()

    # choose loss fn
    criterion = choose_loss_fn(loss_fn, samples_per_cls, no_of_classes, weights_per_cls)

    # training loop
    for epoch in range(1, end_epoch + 1):
        print('\nEpoch {}/{}'.format(epoch, end_epoch))
        print('-' * 20)
        
        # select between training or validation
        for phase in ['Training']:#, 'Validation']:
            if phase == 'Training':
                model.train() # set classification layer to training mode
            else:
                model.eval() # set classification layer to evaluate mode
            
            running_loss = 0.0

            save_train_loss = False
            save_val_loss = False
            save_train_f1 = False
            save_val_f1 = False
            save_train_acc = False
            save_val_acc = False

            # iterate over data.
            for i, batch in enumerate(data_loader[phase], 0):

                # grab data
                inputs, labels = batch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.no_grad():
                    features = conv_net(inputs.to(device))

                with torch.set_grad_enabled(phase=='Training'):
                    outputs = model(features)

                # rare case where final batch in data has size 1 - add leading dimension
                output_size = [int(x) for x in outputs.shape]
                if len(output_size) == 1:
                    outputs = torch.unsqueeze(outputs,0)
                
                # choose loss fn based on function input
                loss = criterion(outputs, labels.to(device))

                # catch NaNs
                if math.isnan(loss):
                    print("\n\nCont'd: got undefined loss (nan)\n\n")
                    del loss, outputs, inputs, labels
                    continue

                pred[phase][ i*batch_size : (batch_size*(i+1)) ] = torch.argmax(outputs, 1).float()
                gt[phase][ i*batch_size : (batch_size*(i+1)) ] = labels

                # backward pass + optimization (only if in training mode)
                if phase == 'Training':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

                string_out = "Epoch [{}/{}]\tStep [{}/{}]\tLoss: {:.5}".format(epoch, end_epoch, i+1, data_length[phase], running_loss / (i+1))
                sys.stdout.write('%s\r' % string_out)
                sys.stdout.flush()

                del loss, outputs, inputs, labels

            # print loss, accuracy, class f1 scores, and harmonic mean f1 for whole epoch
            epoch_loss, accuracy, f1_class_scores = print_scores(running_loss, data_length, phase, gt, pred)

            if phase == 'Training':
                if epoch_loss < min_train_loss:
                    min_train_loss = epoch_loss
                    save_train_loss = True

                if f1_class_scores[1] > max_train_f1:
                    max_train_f1 = f1_class_scores[1]
                    save_train_f1 = True

                if accuracy > max_train_acc:
                    max_train_acc = accuracy
                    save_train_acc = True

            else:
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    save_val_loss = True

                if f1_class_scores[1] > max_val_f1:
                    max_val_f1 = f1_class_scores[1]
                    save_val_f1 = True

                if accuracy > max_val_acc:
                    max_val_acc = accuracy
                    save_val_acc = True

            save_model(model, optimizer, epoch, accuracy, save_train_loss, save_val_loss, save_train_f1, 
                       save_val_f1, save_train_acc, save_val_acc, clf_out, iteration, False)

        scheduler.step()

    del pred_train, labels_train, pred_val, labels_val, pred, gt

    print("Training for iteration {:.0f} finished\n\n".format(iteration))
    return


def fine_tune(train_set, val_set, data_loader, data_length, batch_size, cnn, device, 
              learning_rate, clf_out, loss_fn, samples_per_cls, no_of_classes, weights_per_cls):
    '''
    Function for fine-tuning CNN. 
    
    Inputs:
        train_set [BinaryDataset]: Dataset object for training set
        val_set [BinaryDataset]: Dataset object for validation set
        data_loader [dict]: dict of data loader objects 
            e.g. data_loader = {'Training': train_loader, 'Validation': val_loader}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        data_length [dict]: dict of data loader lengths
            e.g. data_length = {'Training': len(train_loader), 'Validation': len(val_loader)}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        batch_size [int]: batch size in training
        cnn [torch.nn]: CNN, loaded to device
        device [torch.device]: cpu or gpu
            instantiated using torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learning_rate [float]: learning rate applied to backpropagation
        clf_out [str]: path to where model is saved
        loss_fn [str]: specifies loss function to use
        samples_per_cls [np.array(n,)]: array storing count of samples for each of the n classes
        no_of_classes [int]: number of classes 
        weights_per_cls [torch.tensor(n,)]: torch tensor of weights for each of the n classes, loaded onto device
    '''
    # save csv for training and validation sets
    folder_path = clf_out + "1/"
    train_set.df.to_csv(folder_path + "train_set_unique_labels.csv")
    val_set.df.to_csv(folder_path + "val_set_unique_labels.csv")

    # define variables for calculating statistics
    pred_train = torch.zeros(len(train_set)).to(device)
    labels_train = torch.zeros(len(train_set)).to(device)
    pred_val = torch.zeros(len(val_set)).to(device)
    labels_val = torch.zeros(len(val_set)).to(device)
    pred = {'Training': pred_train, 'Validation': pred_val}
    gt = {'Training': labels_train, 'Validation': labels_val}
    min_train_loss = 999999999
    min_val_loss = 999999999
    max_train_f1 = 0
    max_val_f1 = 0
    max_train_acc = 0
    max_val_acc = 0

    # set all layers to trainable
    for param in cnn.parameters():
        param.requires_grad = True

    # setup optimizer and learning rate scheduler
    scheduler_step = 33
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.2)

    # choose loss fn
    criterion = choose_loss_fn(loss_fn, samples_per_cls, no_of_classes, weights_per_cls)

    # training loop
    end_epoch = 101
    for epoch in range(1, end_epoch):
        print('\nEpoch {}/{}'.format(epoch, end_epoch))
        print('-' * 20)
        
        # select between training or validation
        for phase in ['Training', 'Validation']:
            if phase == 'Training':
                cnn.train() # set classification layer to training mode
            else:
                cnn.eval() # set classification layer to evaluate mode
            
            running_loss = 0.0

            save_train_loss = False
            save_val_loss = False
            save_train_f1 = False
            save_val_f1 = False
            save_train_acc = False
            save_val_acc = False

            # iterate over data.
            for i, batch in enumerate(data_loader[phase], 0):

                # grab data
                inputs, labels = batch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase=='Training'):
                    outputs = cnn(inputs.to(device))

                # rare case where final batch in data has size 1 - add leading dimension
                output_size = [int(x) for x in outputs.shape]
                if len(output_size) == 1:
                    outputs = torch.unsqueeze(outputs,0)
                
                # choose loss fn based on function input
                loss = criterion(outputs, labels.to(device))

                # catch NaNs
                if math.isnan(loss):
                    print("\n\nCont'd: got undefined loss (nan)\n\n")
                    del loss, outputs, inputs, labels
                    continue

                pred[phase][ i*batch_size : (batch_size*(i+1)) ] = torch.argmax(outputs, 1).float()
                gt[phase][ i*batch_size : (batch_size*(i+1)) ] = labels

                # backward pass + optimization (only if in training mode)
                if phase == 'Training':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

                string_out = "Epoch [{}/{}]\tStep [{}/{}]\tLoss: {:.5}".format(epoch, end_epoch, i+1, data_length[phase], running_loss / (i+1))
                sys.stdout.write('%s\r' % string_out)
                sys.stdout.flush()

                del loss, outputs, inputs, labels

            # print loss, accuracy, class f1 scores, and harmonic mean f1 for whole epoch
            epoch_loss, accuracy, f1_class_scores = print_scores(running_loss, data_length, phase, gt, pred)

            if phase == 'Training':
                if epoch_loss < min_train_loss:
                    min_train_loss = epoch_loss
                    save_train_loss = True

                if f1_class_scores[1] > max_train_f1:
                    max_train_f1 = f1_class_scores[1]
                    save_train_f1 = True

                if accuracy > max_train_acc:
                    max_train_acc = accuracy
                    save_train_acc = True

            else:
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    save_val_loss = True

                if f1_class_scores[1] > max_val_f1:
                    max_val_f1 = f1_class_scores[1]
                    save_val_f1 = True

                if accuracy > max_val_acc:
                    max_val_acc = accuracy
                    save_val_acc = True

            save_model(cnn, optimizer, epoch, accuracy, save_train_loss, save_val_loss, save_train_f1, 
                       save_val_f1, save_train_acc, save_val_acc, clf_out, None, True)

        scheduler.step()

    del pred_train, labels_train, pred_val, labels_val, pred, gt

    print("Fine tuning complete.\n\n")
    return


def clf_tester(dataset, data_loader, batch_size, conv_net, model, device):
    '''
    Function for classifying data in evaluation mode. Returns boolean array corresponding
    
    Inputs:
        dataset [BinaryDataset]: Dataset object for training set
        data_loader [torch.utils.data.DataLoader]: data loader object for dataset
        batch_size [int]: batch size for evaluation
        conv_net [torch.nn]: feature extractor, loaded to device
        model [torch.nn]: classification layer, loaded to device
        device [torch.device]: cpu or gpu
            instantiated using torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    Outputs:
        positives [np.array(n,)]: array marking the correctly classified samples
            e.g. [1,1,0,1,0,0] means that samples 0, 1, 3 were correctly classified
    '''
    # define variables for calculating statistics
    pred = torch.zeros(len(dataset)).to(device)
    gt = torch.zeros(len(dataset)).to(device)

    # set model to evaluation mode
    conv_net.eval()
    model.eval()

    running_loss = 0.0

    # iterate over data.
    for i, batch in enumerate(data_loader, 0):

        string_out = "Step [{}/{}]".format(i+1, len(data_loader))
        sys.stdout.write('%s\r' % string_out)
        sys.stdout.flush()

        # grab data
        inputs, labels = batch

        # forward pass
        with torch.no_grad():
            features = conv_net(inputs.to(device))
            outputs = model(features)            

        # rare case where final batch in data has size 1 - add leading dimension
        output_size = [int(x) for x in outputs.shape]
        if len(output_size) == 1:
            outputs = torch.unsqueeze(outputs,0)

        pred[ i*batch_size : (batch_size*(i+1)) ] = torch.argmax(outputs, 1).float()
        gt[ i*batch_size : (batch_size*(i+1)) ] = labels

        del outputs, inputs, labels

    # print loss, accuracy, class f1 scores, and harmonic mean f1 for whole epoch
    accuracy = accuracy_score(gt.detach().cpu(), pred.detach().cpu())
    f1_class_scores = f1_score(gt.detach().cpu(), pred.detach().cpu(), average=None, labels=[0,1]).tolist()
    print('Accuracy: {:.4f}, \tF1: [{:.4f}, {:.4f}]'
          .format(accuracy, f1_class_scores[0], f1_class_scores[1]))

    positives = (pred == gt).cpu().numpy()

    del pred, gt

    print("\n\n")
    return positives


def create_meta_data(train_set, val_set, test_set, data_loader, data_length, 
                              batch_size, conv_net, device, data_out, clfs):
    '''
    Function for creating meta-data using logit outputs from each classification layer.
    
    Inputs:
        train_set [BinaryDataset]: Dataset object for training set
        val_set [BinaryDataset]: Dataset object for validation set
        data_loader [dict]: dict of data loader objects 
            e.g. data_loader = {'Training': train_loader, 'Validation': val_loader}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        data_length [dict]: dict of data loader lengths
            e.g. data_length = {'Training': len(train_loader), 'Validation': len(val_loader)}
            where train_loader and val_loader are of type torch.utils.data.DataLoader
        end_epoch [int]: number of epochs desired for training
        batch_size [int]: batch size in training
        conv_net [torch.nn]: feature extractor, loaded to device
        device [torch.device]: cpu or gpu
            instantiated using torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_out [str]: path for saving meta data (saved as type np.array)
        clfs [list]: list where each element is a classification layer (already loaded to device)
    '''
    boost_data_train = torch.zeros(len(train_set), len(clfs) * 2).to(device)
    boost_data_val = torch.zeros(len(val_set), len(clfs) * 2).to(device)
    boost_data_test = torch.zeros(len(test_set), len(clfs) * 2).to(device)
    boost_data = {'Training': boost_data_train, 'Validation': boost_data_val, 'Testing': boost_data_test}

    labels_train = torch.zeros(len(train_set)).to(device)
    labels_val = torch.zeros(len(val_set)).to(device)
    labels_test = torch.zeros(len(test_set)).to(device)
    gt = {'Training': labels_train, 'Validation': labels_val, 'Testing': labels_test}

    conv_net.eval()
    for i in range(len(clfs)):
        clfs[i].eval()
    
    # select between training or validation
    for phase in ['Training', 'Validation', 'Testing']:

        # iterate over data.
        for i, batch in enumerate(data_loader[phase], 0):

            string_out = "Step [{}/{}]".format(i+1, len(data_loader[phase]))
            sys.stdout.write('%s\r' % string_out)
            sys.stdout.flush()

            # grab data
            inputs, labels = batch

            # forward pass
            with torch.no_grad():
                features = conv_net(inputs.to(device)) 
            
            ens_in = torch.randn(0).to(device)
            for j in range(len(clfs)):
                with torch.no_grad():
                    ens_in = torch.cat((ens_in, clfs[j](features)), dim=1)            

            boost_data[phase][ i*batch_size : (batch_size*(i+1)), : ] = ens_in
            gt[phase][ i*batch_size : (batch_size*(i+1)) ] = labels

    # save for offline
    boost_data_train = boost_data["Training"].cpu().detach().numpy()
    boost_data_val = boost_data["Validation"].cpu().detach().numpy()
    boost_data_test = boost_data["Testing"].cpu().detach().numpy()
    np.save(data_out + "boost_data_train.npy", boost_data_train)
    np.save(data_out + "boost_data_val.npy", boost_data_val)
    np.save(data_out + "boost_data_test.npy", boost_data_test)

    boost_labels_train = gt["Training"].cpu().detach().numpy()
    boost_labels_val = gt["Validation"].cpu().detach().numpy()
    boost_labels_test = gt["Testing"].cpu().detach().numpy()
    np.save(data_out + "boost_labels_train.npy", boost_labels_train)
    np.save(data_out + "boost_labels_val.npy", boost_labels_val)
    np.save(data_out + "boost_labels_test.npy", boost_labels_test)

    return


def fn_and_fp(y_test, test_preds):
    # count false positives and false negatives
    fn = 0
    fp = 0
    for i in range(y_test.shape[0]):
        if test_preds[i] == 0 and y_test[i] == 1:
            fn += 1
        elif test_preds[i] == 1 and y_test[i] == 0:
            fp += 1

    print("# false negatives: ", fn)
    print("# false positives: ", fp)
    return

def softmax_transform(X):
    X_softmax = np.empty(X.shape)
    num_clf = int(X.shape[1] / 2)
    for i in range(num_clf):
        X_softmax[:,i:i+2] = softmax(X[:,i:i+2], axis=1)
    return X_softmax

def model_average(X):
    # X needs to be in softmax form
    num_clf = int(X.shape[1] / 2)
    X_logit_avg = np.zeros((X.shape[0],2))
    for i in range(num_clf):
        X_logit_avg[:,0:2] = X_logit_avg[:,0:2] + X[:,i:i+2]
    return np.argmax(X_logit_avg, axis=1)

def majority_voting(X):
    # X needs to be in softmax form
    num_clf = int(X.shape[1] / 2)
    X_votes = np.zeros((X.shape[0],2))
    for i in range(X.shape[0]):
        for j in range(num_clf):
            if X[i,2*j+1] > X[i,2*j]:
                X_votes[i,1] += 1
            else:
                X_votes[i,0] += 1
    return np.argmax(X_votes, axis=1)