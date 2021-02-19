# inovor-ensemble
Ensemble classifier layers for a single deep neural network

## Environment Setup

We recommend starting with creating a Python virtual environment and installing the most recent stable version of PyTorch, e.g.:

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Then install additional requirements using pip:

```bash
pip install -r requirements_pip.txt
```

## Dataset Setup

The primary dataset used for development and testing this classifer ensemble method was the [Airbus Ship Detection dataset](https://www.kaggle.com/c/airbus-ship-detection). This dataset can be downloaded and used directly. The `dl.py` labels image IDs with pixel coordinates as `true` and those without `false`.


## Training a Classifer Ensemble

To start training a classifier ensemble, start `train_clf.py` from the commmand line. An example of the options used to train using the [Airbus Ship Detection dataset](https://www.kaggle.com/c/airbus-ship-detection) with EfficientNet-B0: 

```bash
python train_clfs.py --dir ./airbus-ship-detection/train_v2/ --csv ./airbus-ship-detection/train_ship_segmentations_v2.csv --model_type EFFICIENTNET-B0 --clf_out ./clf_out
```

A similar set of options can be used to train the meta-learner, e.g.:

```bash
python train_meta.py --dir ./airbus-ship-detection/train_v2/ --csv ./airbus-ship-detection/train_ship_segmentations_v2.csv --model_type EFFICIENTNET-B0 --data_in ./clf_out --data_out . --ens_num 10 --ens_type xgboost
```

Then the ensemble can be applied to classify the dataset:

```bash
python train_meta.py --dir ./airbus-ship-detection/train_v2/ --csv ./airbus-ship-detection/train_ship_segmentations_v2.csv --model_type EFFICIENTNET-B0 --data_in ./clf_out --ens_num 10 --ens_type xgboost
```

Use command line option `--help` for more input options for training the classifiers and meta-learner.