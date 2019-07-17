from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence, Normalizer
from keras.models import Sequential
from keras.datasets import imdb
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D, Activation, Lambda, Embedding
from keras.utils import np_utils
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import pandas as pd

import torch #for building NN model
import torchvision
#pandas for reading data from file
import pandas as pd 
#train_test_split for seperaring train, test dataset for validating our model accuracy during train
from sklearn.model_selection import train_test_split 

def get_dataset(task):
	# load the train dataset
	dataset = pd.read_csv(r"PaviaU_5.csv",dtype = np.float32)
	print(dataset.shape)
	dataset.head()

	# get array of labels
	labels = dataset.label.values
	# get all features values in table except label
	features = dataset.loc[:,dataset.columns != "label"].values
	# print(features[0])
	features = features/255 # normalization
	features = torch.Tensor(features)
	labels = torch.Tensor(labels)
	train_dataset = torch.utils.data.TensorDataset(features,labels)

	# load the train dataset
	dataset = pd.read_csv(r"PaviaU.csv",dtype = np.float32)
	print(dataset.shape)
	dataset.head()

	# get array of labels
	labels = dataset.label.values
	# get all features values in table except label
	features = dataset.loc[:,dataset.columns != "label"].values
	# print(features[0])
	features = features/255 # normalization
	features = torch.Tensor(features)
	labels = torch.Tensor(labels)
	test_dataset = torch.utils.data.TensorDataset(features,labels)
	
	return train_dataset,test_dataset