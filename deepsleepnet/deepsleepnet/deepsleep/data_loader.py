import os
import numpy as np
from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_oversample
import re
import sys
sys.path.append('../')
import utils


class NonSeqDataLoader(object):

    def __init__(self, dataFileList, dataset, valFraction = .01): #, n_folds, fold_idx):
        self.dataFileList = dataFileList
        self.dataset = dataset
        self.valFraction = valFraction
        #self.n_folds = n_folds
        #self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_cv_data(self, list_files):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files = None, numChannels = 1, stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5}):
        
        # Set settings according to the dataset being used
        if self.dataset == 'debug':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif self.dataset == 'cicc':
            featureExtractionFunc = utils.extractFromCicc
            possibleChannels = ['C3-M2','C4-M1'][:numChannels]
        elif self.dataset == 'wsc':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif self.dataset == 'shhs':
            featureExtractionFunc = utils.extractFromShhs
            possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
        elif self.dataset == 'SleepProfiler':
            featureExtractionFunc = utils.extractFromSleepProfiler
            possibleChannels = [['EEG3'],['EEG2']][:numChannels]
        elif self.dataset == 'mass':
            featureExtractionFunc = utils.extractFromMass
            possibleChannels = ['C3','C4'][:numChannels]
        elif self.dataset == 'isruc':
            featureExtractionFunc = utils.extractFromIsruc
            possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]
        elif self.dataset == 'mros':
            featureExtractionFunc = utils.extractFromMrOs
            # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
            # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
            possibleChannels = [["C3"],["C4"]][:numChannels]
        elif self.dataset == 'eegbud':
            featureExtractionFunc = utils.extractFromEegBuds
            possibleChannels = [["5","CH5","Ch5"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]
        else:
            print('Unknown dataset')

        # Extract all data
        print("Loading data...")
        ptList = featureExtractionFunc(dataFileList = self.dataFileList,downSampleTo = 100, possibleChannels = possibleChannels, stages = stages, extractFeatures=utils.getRawSignal)
        print("Data loaded.")

        # Separate subjects for early stopping
        ptIdList = list(set([pt['patientId'] for pt in ptList]))
        valPtIdList = ptIdList[::int(1/self.valFraction)] # list of patients to use in early stopping
        valPts = [pt for pt in ptList if pt['patientId'] in valPtIdList]
        trainPts = [pt for pt in ptList if pt['patientId'] not in valPtIdList]
        
        # Separate data and labels
        data_train = np.vstack([pt['features'] for pt in trainPts])
        label_train = np.hstack([pt['labels'] for pt in trainPts])
        data_val = np.vstack([pt['features'] for pt in valPts])
        label_val = np.hstack([pt['labels'] for pt in valPts])

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Remove unwanted samples
        data_train = np.delete(data_train,np.where(label_train == stages['undefined']),axis=0)
        label_train = np.delete(label_train,np.where(label_train == stages['undefined']),axis=0)
        data_val = np.delete(data_val,np.where(label_val == stages['undefined']),axis=0)
        label_val = np.delete(label_val,np.where(label_val == stages['undefined']),axis=0)

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        print("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train)
        print(" ")
        print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val)
        print(" ")

        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train)
        print(" ")

        return x_train, y_train, data_val, label_val

class SeqDataLoader(object):

    def __init__(self, dataFileList, dataset, valFraction = .01): #, n_folds, fold_idx):
        self.dataFileList = dataFileList
        self.dataset = dataset
        self.valFraction = valFraction
        #self.n_folds = n_folds
        #self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            
            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
            
            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]
            
            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)
            
            data.append(tmp_data)
            labels.append(tmp_labels)
            
        return data, labels

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files = None, numChannels = 1, stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5}):
        
        # Set settings according to the dataset being used
        if self.dataset == 'debug':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif self.dataset == 'cicc':
            featureExtractionFunc = utils.extractFromCicc
            possibleChannels = ['C3-M2','C4-M1'][:numChannels]
        elif self.dataset == 'wsc':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif self.dataset == 'shhs':
            featureExtractionFunc = utils.extractFromShhs
            possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
        elif self.dataset == 'SleepProfiler':
            featureExtractionFunc = utils.extractFromSleepProfiler
            possibleChannels = [['EEG3'],['EEG2']][:numChannels]
        elif self.dataset == 'mass':
            featureExtractionFunc = utils.extractFromMass
            possibleChannels = ['C3','C4'][:numChannels]
        elif self.dataset == 'isruc':
            featureExtractionFunc = utils.extractFromIsruc
            possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]
        elif self.dataset == 'mros':
            featureExtractionFunc = utils.extractFromMrOs
            # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
            # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
            possibleChannels = [["C3"],["C4"]][:numChannels]
        elif self.dataset == 'eegbud':
            featureExtractionFunc = utils.extractFromEegBuds
            possibleChannels = [["5","CH5","Ch5"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]
        else:
            print('Unknown dataset')

        # Extract all data
        print("Loading data...")
        ptList = featureExtractionFunc(dataFileList = self.dataFileList,downSampleTo = 100, possibleChannels = possibleChannels, stages = stages, extractFeatures=utils.getRawSignal)
        print("Data loaded.")

        # Separate subjects for early stopping
        ptIdList = list(set([pt['patientId'] for pt in ptList]))
        valPtIdList = ptIdList[::int(1/self.valFraction)] # list of patients to use in early stopping
        valPts = [pt for pt in ptList if pt['patientId'] in valPtIdList]
        trainPts = [pt for pt in ptList if pt['patientId'] not in valPtIdList]

        # Separate data and labels, removing undefined epochs as we go
        data_train = [np.delete(np.array(pt['features']),np.where(pt['labels'] == stages['undefined']),axis=0)[:,:,None] for pt in trainPts]
        label_train = [np.delete(np.array(pt['labels']),np.where(pt['labels'] == stages['undefined']),axis=0) for pt in trainPts]
        data_val = [np.delete(np.array(pt['features']),np.where(pt['labels'] == stages['undefined']),axis=0)[:,:,None] for pt in valPts]
        label_val = [np.delete(np.array(pt['labels']),np.where(pt['labels'] == stages['undefined']),axis=0) for pt in valPts]
        print("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            #print(d.shape)
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        #print_n_samples_each_class(np.hstack(label_train))
        print(" ")
        print("Validation set: n_subjects={}".format(len(data_val)))
        n_valid_examples = 0
        for d in data_val:
            #print(d.shape)
            n_valid_examples += d.shape[0]
        print("Number of examples = {}".format(n_valid_examples))
        #print_n_samples_each_class(np.hstack(label_val))
        print(" ")

        return data_train, label_train, data_val, label_val

    @staticmethod
    def load_subject_data(dataFile, dataset, numChannels = 1, stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5}):
        
        # Set settings according to the dataset being used
        if dataset == 'debug':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif dataset == 'cicc':
            featureExtractionFunc = utils.extractFromCicc
            possibleChannels = ['C3-M2','C4-M1'][:numChannels]
        elif dataset == 'wsc':
            featureExtractionFunc = utils.extractFromWsc
            possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        elif dataset == 'shhs':
            featureExtractionFunc = utils.extractFromShhs
            possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
        elif dataset == 'SleepProfiler':
            featureExtractionFunc = utils.extractFromSleepProfiler
            possibleChannels = [['EEG3'],['EEG2']][:numChannels]
        elif dataset == 'mass':
            featureExtractionFunc = utils.extractFromMass
            possibleChannels = ['C3','C4'][:numChannels]
        elif dataset == 'isruc':
            featureExtractionFunc = utils.extractFromIsruc
            possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]
        elif dataset == 'mros':
            featureExtractionFunc = utils.extractFromMrOs
            # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
            # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
            possibleChannels = [["C3"],["C4"]][:numChannels]
        elif dataset == 'eegbud':
            featureExtractionFunc = utils.extractFromEegBuds
            possibleChannels = [["5","CH5","Ch5"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]
        else:
            print('Unknown dataset')

        # Extract all data
        print("Loading data...")
        pt = featureExtractionFunc(dataFileList = [dataFile],downSampleTo = 100, possibleChannels = possibleChannels, stages = stages, extractFeatures=utils.getRawSignal)[0]

        # Separate data and labels, removing undefined epochs as we go
        data = np.delete(np.array(pt['features']),np.where(pt['labels'] == stages['undefined']),axis=0)[:,:,None,None]
        labels = np.delete(np.array(pt['labels']),np.where(pt['labels'] == stages['undefined']),axis=0) 

        return [data], [labels]

