import sys
import os
import datetime
import h5py
import csv
import pandas as pd
import itertools
import multiprocessing
import tensorflow.keras as keras
import pyedflib.highlevel as edf
import xml.etree.ElementTree as ET
import numpy as np
import statistics as s
import tensorflow as tf
import random
from dateutil import parser
from collections import Counter
from glob import glob
from scipy.io import loadmat
from scipy.stats import mode
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from scipy import signal as sg
from scipy.stats import zscore
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from os import path

def splitDataNonSequential(ptList,validationList,stages,earlyStoppingFraction=.1,omitTrainArtifacts = False,omitValidArtifacts = False,maxRecordings = None,augment = False):
    # Separate patients into training, validationa and early stopping sets
    # Returns tensorflow dataset objects in order (training dataset,early stopping dataset, validation dataset)
    # 
    # INPUTS
    # ptList ---------------------------- (dictionary List) List of dictionaries containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    # validationList -------------------- (int List) List of indices of the patients in ptList to hold out for validation.
    # stages ---------------------------- (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                      label (int)
    # earlyStoppingFraction ------------- (float) Fraction of patients to use in early stopping
    # 
    # OUTPUTS
    # trainDataset ---------------------- (dataset) Dataset to use for training
    # earlyStoppingDataset -------------- (dataset) Dataset to use for early stopping
    # validationDataset ----------------- (dataset) Dataset to hold out for validation
    
    
    # Obtain train-validation split on remaining patients
    validationPts = [ptList[pt] for pt in validationList]
    [print(ptList[pt]['patientId']) for pt in validationList]
    validationLabels = np.concatenate([pt['labels'] for pt in validationPts]) if validationPts else np.array([])
    validationData = np.concatenate([pt['features'] for pt in validationPts]) if validationPts else np.array([])

    # Relabel regions of validation data with artifacts as 'undefined', if specified
    if validationPts and omitValidArtifacts and 'containsArtifact' in validationPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in validationPts])
        validationLabels[containsArtifact] = stages['undefined']

    del validationPts
    
    trainingPts = [ptList[pt] for pt in set(range(0,len(ptList))) - set(validationList)]
    if maxRecordings is not None:
        ptIdList = np.unique(np.array([pt['patientId'] for pt in trainingPts]))
        ptRecordingsCount = {ptId: 0 for ptId in ptIdList}
        
        # Count number of recordings each patient has and remove recordings exceeding the limit
        for i, pt in enumerate(trainingPts):
            ptRecordingsCount[pt['patientId']] += 1
            if ptRecordingsCount[pt['patientId']] > maxRecordings:
                del trainingPts[i]
        
    trainingLabels = np.concatenate([pt['labels'] for pt in trainingPts])
    trainingData = np.concatenate([pt['features'] for pt in trainingPts])
    
    # Relabel regions of training data with artifacts as 'undefined', if specified
    if omitTrainArtifacts and 'containsArtifact' in trainingPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in trainingPts])
        trainingLabels[containsArtifact] = stages['undefined']
    del trainingPts
    
    # Remove undefined epochs
    trainingData = np.delete(trainingData,np.where(trainingLabels == stages['undefined']),axis=0)
    trainingLabels = np.delete(trainingLabels,np.where(trainingLabels == stages['undefined']))
    validationData = np.delete(validationData,np.where(validationLabels == stages['undefined']),axis=0)
    validationLabels = np.delete(validationLabels,np.where(validationLabels == stages['undefined']))
    
    # Obtain Class Weights
    possibleLabels = np.unique(trainingLabels)
    weights = class_weight.compute_class_weight('balanced',
                                               possibleLabels,
                                               trainingLabels)
    class_weights = {possibleLabels[0]:weights[0],possibleLabels[1]:weights[1],possibleLabels[2]:weights[2],possibleLabels[3]:weights[3],possibleLabels[4]:weights[4]}
    
    # Randomly separate out early stopping set
    trainingData[np.where(np.isnan(trainingData))] = 0
    x_train, x_earlyStopping, y_train, y_earlyStopping = train_test_split(trainingData,trainingLabels,test_size=earlyStoppingFraction,random_state=0,stratify=trainingLabels)
    del trainingData, trainingLabels
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train,to_categorical(y_train))).batch(32)

    # Augment training dataset by adding random white noise, if specified
    if augment:
        print('Augmenting training data with gaussian white noise')
        noiseAddition = lambda x,y: (tf.add(x,tf.random.normal(shape=x.shape, mean=0.0, stddev=0.00001, dtype=tf.float16)),y)
        trainDataset = trainDataset.unbatch()
        trainDataset = trainDataset.concatenate(trainDataset.map(noiseAddition)).concatenate(trainDataset.map(noiseAddition)).batch(32)
    
    del x_train, y_train
    earlyStoppingDataset = tf.data.Dataset.from_tensor_slices((x_earlyStopping,to_categorical(y_earlyStopping))).batch(32)
    del x_earlyStopping, y_earlyStopping
    validationDataset = tf.data.Dataset.from_tensor_slices((validationData,to_categorical(validationLabels))).batch(32) if validationData.size is not 0 else tf.data.Dataset.from_tensor_slices((validationData))
    
    return (trainDataset,earlyStoppingDataset,validationDataset,class_weights)

def splitDataSequential(ptList,validationList,stages,earlyStoppingFraction=.1,omitTrainArtifacts = False,omitValidArtifacts = False,augment = False):
    # Separate patient into training, validation and early stopping sets while keeping from same subjects in sequence.
    # Returns tensorflow dataset objects in order (training dataset,early stopping dataset, validation dataset)
    # 
    # INPUTS
    # ptList ---------------------------- (dictionary List) List of dictionaries containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    # validationList -------------------- (int List) List of indices of the patients in ptList to hold out for validation.
    # stages ---------------------------- (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                      label (int)
    # earlyStoppingFraction ------------- (float) Fraction of patients to use in early stopping
    # 
    # OUTPUTS
    # trainDataset ---------------------- (dataset) Dataset to use for training
    # earlyStoppingDataset -------------- (dataset) Dataset to use for early stopping
    # validationDataset ----------------- (dataset) Dataset to hold out for validation
    
    maxLength = np.max([pt['labels'].size for pt in ptList]) # Get number of samples in longest recording
    makeSameLength = lambda x,c: np.pad(x,((0,maxLength - x.shape[0]),)+(x.ndim-1)*((0,0),),constant_values=c) # Extends an array to desired length by adding some constant to end

    # Obtain train-validation split on remaining patients
    validationLabels = np.array([makeSameLength(ptList[pt]['labels'],stages['undefined']) for pt in validationList])
    validationData = np.array([makeSameLength(ptList[pt]['features'],.1) for pt in validationList])

    # Relabel regions of validation data with artifacts as 'undefined', if specified
    if omitValidArtifacts and 'containsArtifact' in validationPts[0].keys():
        containsArtifact = np.array([makeSameLength(ptList[pt]['containsArtifact'],False) for pt in validationList])
        validationLabels[containsArtifact] = stages['undefined']

    trainingLabels = np.array([makeSameLength(ptList[pt]['labels'],stages['undefined']) for pt in set(range(0,len(ptList))) - set(validationList)])
    trainingData = np.array([makeSameLength(ptList[pt]['features'],.1) for pt in set(range(0,len(ptList))) - set(validationList)])

    # Relabel regions of training data with artifacts as 'undefined', if specified
    if omitValidArtifacts and 'containsArtifact' in validationPts[0].keys():
        containsArtifact = np.array([makeSameLength(ptList[pt]['containsArtifact'],False) for pt in set(range(0,len(ptList))) - set(validationList)])
        trainingLabels[containsArtifact] = stages['undefined']

    # Obtain Class Weights
    weightLabels = trainingLabels.flatten()
    weightLabels = weightLabels[weightLabels != stages['undefined']] # Exclude undefined samples from weight calculation
    possibleLabels = np.unique(weightLabels)
    weights = class_weight.compute_class_weight('balanced',
                                                possibleLabels,
                                                weightLabels)
    class_weights = {possibleLabels[0]:weights[0],
                     possibleLabels[1]:weights[1],
                     possibleLabels[2]:weights[2],
                     possibleLabels[3]:weights[3],
                     possibleLabels[4]:weights[4],
                     stages['undefined']:0.0}
    
    # Separate out fraction of patients for early stopping set
    earlyStoppingPts = slice(0,trainingData.shape[0],np.int(earlyStoppingFraction**-1))
    x_earlyStopping = trainingData[earlyStoppingPts]
    y_earlyStopping = trainingLabels[earlyStoppingPts]
    x_train = np.delete(trainingData,earlyStoppingPts,axis=0)
    y_train = np.delete(trainingLabels,earlyStoppingPts,axis=0)
    del trainingData, trainingLabels # Memory management
    
    # Weight every stage equally except 'undefined', which will have a weight of 0 so that it isn't counted
    trainingWeights = np.ones(y_train.shape)
    trainingWeights[y_train == stages['undefined']] = 0
    earlyStoppingWeights = np.ones(y_earlyStopping.shape)*.2
    earlyStoppingWeights[y_earlyStopping == stages['undefined']] = 0
    validationWeights = np.ones(validationLabels.shape)*.2
    validationWeights[validationLabels == stages['undefined']] = 0

    # Change 'undefined' labels to something else so that the 'unknown' class is not counted when computing the number of classes
    y_earlyStopping[y_earlyStopping == stages['undefined']] = stages['awake'] # Choice of relabeling arbitrary
    y_train[y_train == stages['undefined']] = stages['awake']
    
    # Wrap in dataset objects
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train,to_categorical(y_train),trainingWeights)).batch(1)
    del x_train, y_train
    
    # Augment training dataset by adding random white noise, if specified
    if augment:
        print('Augmenting training data with gaussian white noise')
        noiseAddition = lambda x,y: (tf.add(x,tf.random.normal(shape=x.shape, mean=0.0, stddev=0.00001, dtype=tf.float16)),y)
        trainDataset = trainDataset.unbatch()
        trainDataset = trainDataset.concatenate(trainDataset.map(noiseAddition)).concatenate(trainDataset.map(noiseAddition)).batch(32)

    earlyStoppingDataset = tf.data.Dataset.from_tensor_slices((x_earlyStopping,to_categorical(y_earlyStopping),earlyStoppingWeights)).batch(1)
    del x_earlyStopping, y_earlyStopping
    validationDataset = tf.data.Dataset.from_tensor_slices((validationData,to_categorical(validationLabels),validationWeights)).batch(1)
    
    return (trainDataset,earlyStoppingDataset,validationDataset,weights)

def advInfSplitDataNonSequential(srcPtList,targPtList,validationList,stages,earlyStoppingFraction=.1,omitTrainArtifacts = False,omitValidArtifacts = False,maxRecordings = None,augment = False):
    # Separate patients into training, validationa and early stopping sets
    # Returns tensorflow dataset objects in order (training dataset,early stopping dataset, validation dataset)
    # 
    # INPUTS
    # ptList ---------------------------- (dictionary List) List of dictionaries containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    # validationList -------------------- (int List) List of indices of the patients in ptList to hold out for validation.
    # stages ---------------------------- (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                      label (int)
    # earlyStoppingFraction ------------- (float) Fraction of patients to use in early stopping
    # 
    # OUTPUTS
    # trainDataset ---------------------- (dataset) Dataset to use for training
    # earlyStoppingDataset -------------- (dataset) Dataset to use for early stopping
    # validationDataset ----------------- (dataset) Dataset to hold out for validation
    
    
    # Obtain train-validation split on remaining patients
    validationPts = [targPtList[pt] for pt in validationList]
    [print(targPtList[pt]['patientId']) for pt in validationList]
    validationLabels = np.concatenate([pt['labels'] for pt in validationPts])
    validationData = np.concatenate([pt['features'] for pt in validationPts])

    # Relabel regions of validation data with artifacts as 'undefined', if specified
    if omitValidArtifacts and 'containsArtifact' in validationPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in validationPts])
        validationLabels[containsArtifact] = stages['undefined']

    del validationPts
    
    targTrainingPts = [targPtList[pt] for pt in set(range(0,len(targPtList))) - set(validationList)]
    srcTrainingPts = [srcPtList[pt] for pt in range(len(srcPtList))]
    if maxRecordings is not None:

        # Count number of recordings each target patient has and remove recordings exceeding the limit
        targPtIdList = np.unique(np.array([pt['patientId'] for pt in targTrainingPts]))
        ptRecordingsCount = {ptId: 0 for ptId in targPtIdList}
        for i, pt in enumerate(targTrainingPts):
            ptRecordingsCount[pt['patientId']] += 1
            if ptRecordingsCount[pt['patientId']] > maxRecordings:
                del targTrainingPts[i]

        # Repeat for source patients
        srcPtIdList = np.unique(np.array([pt['patientId'] for pt in srcTrainingPts]))
        ptRecordingsCount = {ptId: 0 for ptId in srcPtIdList}
        for i, pt in enumerate(srcTrainingPts):
            ptRecordingsCount[pt['patientId']] += 1
            if ptRecordingsCount[pt['patientId']] > maxRecordings:
                del srcTrainingPts[i]

        
    targTrainingLabels = np.concatenate([pt['labels'] for pt in targTrainingPts])
    srcTrainingLabels = np.concatenate([pt['labels'] for pt in srcTrainingPts])
    targTrainingData = np.concatenate([pt['features'] for pt in targTrainingPts])
    srcTrainingData = np.concatenate([pt['features'] for pt in srcTrainingPts])

    # Relabel regions of training data with artifacts as 'undefined', if specified
    if omitTrainArtifacts and 'containsArtifact' in targTrainingPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in targTrainingPts])
        targTrainingLabels[containsArtifact] = stages['undefined']
    if omitTrainArtifacts and 'containsArtifact' in srcTrainingPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in srcTrainingPts])
        srcTrainingLabels[containsArtifact] = stages['undefined']
    del targTrainingPts, srcTrainingPts
    
    # Remove undefined epochs
    targTrainingData = np.delete(targTrainingData,np.where(targTrainingLabels == stages['undefined']),axis=0)
    targTrainingLabels = np.delete(targTrainingLabels,np.where(targTrainingLabels == stages['undefined']))
    srcTrainingData = np.delete(srcTrainingData,np.where(srcTrainingLabels == stages['undefined']),axis=0)
    srcTrainingLabels = np.delete(srcTrainingLabels,np.where(srcTrainingLabels == stages['undefined']))
    validationData = np.delete(validationData,np.where(validationLabels == stages['undefined']),axis=0)
    validationLabels = np.delete(validationLabels,np.where(validationLabels == stages['undefined']))
    
    # Randomly separate out early stopping set
    targTrainingData[np.where(np.isnan(targTrainingData))] = 0
    x_train, x_earlyStopping, y_train, y_earlyStopping = train_test_split(targTrainingData,targTrainingLabels,test_size=earlyStoppingFraction,random_state=0,stratify=targTrainingLabels)
    del targTrainingData, targTrainingLabels

    # Create separate label sets for classifier and descriminator
    sampleWeights = np.concatenate((np.ones(y_train.shape)/y_train.shape[0],np.ones(srcTrainingLabels.shape)/srcTrainingLabels.shape[0])) # Weight both datasets evently
    shuffledOrder = np.random.permutation(sampleWeights.shape[0])
    classifierLabels = to_categorical(np.concatenate((y_train,srcTrainingLabels)))[shuffledOrder]
    descriminatorLabels = to_categorical(np.concatenate((np.zeros(y_train.shape),np.ones(srcTrainingLabels.shape))))[shuffledOrder]

    # Wrap everything in dataset object
    trainDataset = tf.data.Dataset.from_tensor_slices((np.concatenate((x_train,srcTrainingData)),
                                                       {"classifier":classifierLabels,"descriminator":descriminatorLabels},
                                                       sampleWeights))
    del x_train, y_train, srcTrainingData, srcTrainingLabels, classifierLabels, descriminatorLabels, sampleWeights, shuffledOrder

    # Shuffle source and target samples
    trainDataset = trainDataset.batch(256) \
                               .shuffle(1000) \
                               .unbatch() \
                               .shuffle(1000) \
                               .batch(32)
    
    # Augment training dataset by adding random white noise, if specified
    if augment:
        print('Augmenting training data with gaussian white noise')
        noiseAddition = lambda x,y: (tf.add(x,tf.random.normal(shape=x.shape, mean=0.0, stddev=0.00001, dtype=tf.float16)),y)
        trainDataset = trainDataset.unbatch()
        trainDataset = trainDataset.concatenate(trainDataset.map(noiseAddition)).concatenate(trainDataset.map(noiseAddition)).batch(32)

    earlyStoppingDataset = tf.data.Dataset.from_tensor_slices((x_earlyStopping,{"classifier":to_categorical(y_earlyStopping)})).batch(32)
    del x_earlyStopping, y_earlyStopping
    validationDataset = tf.data.Dataset.from_tensor_slices((validationData,to_categorical(validationLabels))).batch(32)
    
    return (trainDataset,earlyStoppingDataset,validationDataset)

def unsupSplitDataNonSequential(ptList,validationList,stages,earlyStoppingFraction=.1,omitTrainArtifacts = False,omitValidArtifacts = False,maxRecordings = None,augment = False,unlabeledData = None):
    # Separate patients into training, validationa and early stopping sets
    # Returns tensorflow dataset objects in order (training dataset,early stopping dataset, validation dataset)
    # 
    # INPUTS
    # ptList ---------------------------- (dictionary List) List of dictionaries containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    # validationList -------------------- (int List) List of indices of the patients in ptList to hold out for validation.
    # stages ---------------------------- (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                      label (int)
    # earlyStoppingFraction ------------- (float) Fraction of patients to use in early stopping
    # 
    # OUTPUTS
    # trainDataset ---------------------- (dataset) Dataset to use for training
    # earlyStoppingDataset -------------- (dataset) Dataset to use for early stopping
    
    
    trainingPts = [ptList[pt] for pt in set(range(0,len(ptList))) - set(validationList)]
    if maxRecordings is not None:
        ptIdList = np.unique(np.array([pt['patientId'] for pt in trainingPts]))
        ptRecordingsCount = {ptId: 0 for ptId in ptIdList}

        # Count number of recordings each patient has and remove recordings exceeding the limit
        for i, pt in enumerate(trainingPts):
            ptRecordingsCount[pt['patientId']] += 1
            if ptRecordingsCount[pt['patientId']] > maxRecordings:
                del trainingPts[i]

    # Add unlabeled data, if any
    if unlabeledData is not None: trainingPts = trainingPts + unlabeledData
    
    trainingData = np.concatenate([pt['features'] for pt in trainingPts])
    
    # Randomly separate out early stopping set
    trainingData[np.where(np.isnan(trainingData))] = 0
    x_train, x_earlyStopping = train_test_split(trainingData,test_size=earlyStoppingFraction,random_state=0)
    del trainingData
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train,x_train)).batch(32)
    
    # Augment training dataset by adding random white noise, if specified
    if augment:
        print('Augmenting training data with gaussian white noise')
        noiseAddition = lambda x,y: (tf.add(x,tf.random.normal(shape=x.shape, mean=0.0, stddev=0.00001, dtype=tf.float16)),y)
        trainDataset = trainDataset.unbatch()
        trainDataset = trainDataset.concatenate(trainDataset.map(noiseAddition)).concatenate(trainDataset.map(noiseAddition)).batch(32)

    del x_train
    earlyStoppingDataset = tf.data.Dataset.from_tensor_slices((x_earlyStopping,x_earlyStopping)).batch(32)
    
    return (trainDataset,earlyStoppingDataset)

def splitMultiInputDataNonSequential(ptList,validationList,stages,earlyStoppingFraction=.1,omitTrainArtifacts = False,omitValidArtifacts = False,maxRecordings = None):
    # Separate patients into training, validationa and early stopping sets when using both stft and covariance inputs
    # Returns tensorflow dataset objects in order (training dataset,early stopping dataset, validation dataset)
    # 
    # INPUTS
    # ptList ---------------------------- (dictionary List) List of dictionaries containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    # validationList -------------------- (int List) List of indices of the patients in ptList to hold out for validation.
    # stages ---------------------------- (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                      label (int)
    # earlyStoppingFraction ------------- (float) Fraction of patients to use in early stopping
    # 
    # OUTPUTS
    # trainDataset ---------------------- (dataset) Dataset to use for training
    # earlyStoppingDataset -------------- (dataset) Dataset to use for early stopping
    # validationDataset ----------------- (dataset) Dataset to hold out for validation
    
    # Obtain train-validation split on remaining patients
    validationPts = [ptList[pt] for pt in validationList]
    [print(ptList[pt]['patientId']) for pt in validationList]
    validationLabels = np.concatenate([pt['labels'] for pt in validationPts])
    validationStft = np.concatenate([pt['features']['stft'] for pt in validationPts])
    validationCov = np.concatenate([pt['features']['covariance'] for pt in validationPts])
    validationCov = validationCov[:,np.triu_indices(validationCov.shape[1])] # Extract upper triangular portion

    # Relabel regions of validation data with artifacts as 'undefined', if specified
    if omitValidArtifacts and 'containsArtifact' in validationPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in validationPts])
        validationLabels[containsArtifact] = stages['undefined']

    del validationPts
    
    trainingPts = [ptList[pt] for pt in set(range(0,len(ptList))) - set(validationList)]
    if maxRecordings is not None:
        ptIdList = np.unique(np.array([pt['patientId'] for pt in trainingPts]))
        ptRecordingsCount = {ptId: 0 for ptId in ptIdList}

        # Count number of recordings each patient has and remove recordings exceeding the limit
        for i, pt in enumerate(trainingPts):
            ptRecordingsCount[pt['patientId']] += 1
            if ptRecordingsCount[pt['patientId']] > maxRecordings:
                del trainingPts[i]
        
    trainingLabels = np.concatenate([pt['labels'] for pt in trainingPts])
    trainingStft = np.concatenate([pt['features']['stft'] for pt in trainingPts])
    trainingCov = np.concatenate([pt['features']['covariance'] for pt in trainingPts])
    trainingCov = trainingCov[:,np.triu_indices(trainingCov.shape[1])] # Extract upper triangular portion

    # Relabel regions of training data with artifacts as 'undefined', if specified
    if omitTrainArtifacts and 'containsArtifact' in trainingPts[0].keys():
        containsArtifact = np.concatenate([pt['containsArtifact'] for pt in trainingPts])
        trainingLabels[containsArtifact] = stages['undefined']
    del trainingPts        
    
    # Remove undefined epochs
    trainingStft = np.delete(trainingStft,np.where(trainingLabels == stages['undefined']),axis=0)
    trainingCov = np.delete(trainingCov,np.where(trainingLabels == stages['undefined']),axis=0)
    trainingLabels = np.delete(trainingLabels,np.where(trainingLabels == stages['undefined']))
    validationStft = np.delete(validationStft,np.where(validationLabels == stages['undefined']),axis=0)
    validationCov = np.delete(validationCov,np.where(validationLabels == stages['undefined']),axis=0)
    validationLabels = np.delete(validationLabels,np.where(validationLabels == stages['undefined']))
    
    # Obtain Class Weights
    possibleLabels = np.unique(trainingLabels)
    weights = class_weight.compute_class_weight('balanced',
                                               possibleLabels,
                                               trainingLabels)
    class_weights = {possibleLabels[0]:weights[0],possibleLabels[1]:weights[1],possibleLabels[2]:weights[2],possibleLabels[3]:weights[3],possibleLabels[4]:weights[4]}
    
    # Randomly separate out early stopping set
    trainingStft[np.where(np.isnan(trainingStft))] = 0
    trainingCov[np.where(np.isnan(trainingCov))] = 0
    stft_train, stft_earlyStopping, cov_train, cov_earlyStopping, y_train, y_earlyStopping = train_test_split(trainingStft,trainingCov,trainingLabels,test_size=earlyStoppingFraction,random_state=0,stratify=trainingLabels)
    del trainingStft, trainingCov, trainingLabels
    trainDataset = tf.data.Dataset.from_tensor_slices(({'stft':stft_train,'covariance':cov_train},to_categorical(y_train))).batch(32)
    del stft_train, cov_train, y_train
    earlyStoppingDataset = tf.data.Dataset.from_tensor_slices(({'stft':stft_earlyStopping,'covariance':cov_earlyStopping},to_categorical(y_earlyStopping))).batch(32)
    del stft_earlyStopping, cov_earlyStopping, y_earlyStopping
    validationDataset = tf.data.Dataset.from_tensor_slices(({'stft':validationStft,'covariance':validationCov},to_categorical(validationLabels))).batch(32)
    
    return (trainDataset,earlyStoppingDataset,validationDataset,class_weights)


def transferDatasetAssembler(data,targetData,validationList,stages,**kwargs):
    # Helper function for assembling source and target datasets when using a domain adaptation network

    augment = kwargs['augment'] if 'augment' in kwargs else False
    
    # Create train/earlystopping/validation tensorflow Dataset objects differently
    # depending on whether or not model being trained contains RNN
    if (('sequential' in kwargs) and kwargs['sequential']):
        # Using RNN, and performing transfer learning (not yet implemented)
        raise Exception('This setting not yet implemented')
        
    elif (('sequential' not in kwargs) or not kwargs['sequential']):
        # No RNN, perform transfer learning
        
        # Load source dataset and labels
        sourceData = np.concatenate([pt['features'] for pt in data])
        sourceLabels = np.concatenate([pt['labels'] for pt in data])
        
        # Remove undefined samples
        sourceFeatures = np.delete(sourceData,np.where(sourceLabels == stages['undefined']),axis=0)
        sourceLabels = np.delete(sourceLabels,np.where(sourceLabels == stages['undefined']))
        
        # Wrap in tensorflow dataset object
        sourceDataset = tf.data.Dataset.from_tensor_slices((sourceFeatures,to_categorical(sourceLabels))).batch(32)
        del sourceFeatures,sourceLabels
        
        # Get train/early stopping/validation datasets for target dataset
        targetDataset,earlyStoppingDataset,validationDataset,class_weights = splitDataNonSequential(targetData,validationList,stages,omitTrainArtifacts = False,omitValidArtifacts = False,maxRecordings = None,augment = augment)

    return (sourceDataset,targetDataset,earlyStoppingDataset,validationDataset,class_weights)

def advInfDatasetAssembler(sourceData,targetData,validationList,stages,**kwargs):
    # Helper function for assembling source and target datasets when doing transfer learning

    augment = kwargs['augment'] if 'augment' in kwargs else False
    
    # Create train/earlystopping/validation tensorflow Dataset objects differently
    # depending on whether or not model being trained contains RNN
    if (('sequential' in kwargs) and kwargs['sequential']):
        # Using RNN, and performing transfer learning (not yet implemented)
        raise Exception('This setting not yet implemented')
        
    elif (('sequential' not in kwargs) or not kwargs['sequential']):
        # No RNN, perform transfer learning
        trainDataset,earlyStoppingDataset,validationDataset = advInfSplitDataNonSequential(sourceData,targetData,validationList,stages,maxRecordings = 2,augment = augment)
        
    return (trainDataset,earlyStoppingDataset,validationDataset,None)

def datasetAssembler(data,validationList,stages,**kwargs):
    # Helper function for assembling sequential or non-sequential dataset object depending on settings

    augment = kwargs['augment'] if 'augment' in kwargs else False
    
    # Create train/earlystopping/validation tensorflow Dataset objects differently
    # depending on whether or not model being trained contains RNN
    if (('sequential' in kwargs) and kwargs['sequential']):
        # If using RNN without transfer learning
        trainDataset,earlyStoppingDataset,validationDataset,class_weights = splitDataSequential(data,validationList,stages,augment = augment)
            
    elif (('sequential' not in kwargs) or not kwargs['sequential']) and not kwargs['transfer']:
        # No RNN, no transfer
        trainDataset,earlyStoppingDataset,validationDataset,class_weights = splitDataNonSequential(data,validationList,stages,maxRecordings = 2,augment = augment)
    return (trainDataset,earlyStoppingDataset,validationDataset,class_weights)

def unsupDatasetAssembler(data,validationList,stages,unlabeledData = None,**kwargs):
    # Helper function for assembling sequential or non-sequential dataset object for use in unsupervised task

    augment = kwargs['augment'] if 'augment' in kwargs else False
    
    # Create train/earlystopping tensorflow Dataset objects differently
    # depending on whether or not model being trained contains RNN
    if (('sequential' in kwargs) and kwargs['sequential']):
        # If using RNN in unsupervised task
        raise Exception('This setting not yet implemented')
            
    elif ('sequential' not in kwargs) or not kwargs['sequential']:
        # No RNN, unsupervised task
        trainDataset,earlyStoppingDataset = unsupSplitDataNonSequential(data,validationList,stages,maxRecordings = 2,augment = augment,unlabeledData = unlabeledData)
    return (trainDataset,earlyStoppingDataset)

def datasetMultiInputAssembler(data,validationList,stages,**kwargs):
    # Helper function for assembling sequential or non-sequential dataset object depending on settings

    # Create train/earlystopping/validation tensorflow Dataset objects differently
    # depending on whether or not model being trained contains RNN
    if (('sequential' in kwargs) and kwargs['sequential']):
        # If using RNN
        raise Exception('This setting not yet implemented')
    
    elif (('sequential' not in kwargs) or not kwargs['sequential']) and not kwargs['transfer']:
        # No RNN, no transfer
        trainDataset,earlyStoppingDataset,validationDataset,class_weights = splitMultiInputDataNonSequential(data,validationList,stages,maxRecordings = 2)
    return (trainDataset,earlyStoppingDataset,validationDataset,class_weights)

def getNumBatches(dataset):
    # Get total number of batches in a dataset

    batchesCount = 0
    for _,*_ in dataset:
        # Tally batches in dataset
        batchesCount = batchesCount + 1

    return batchesCount

def dataset2Numpy(dataset):
    # Obtains the dataset features and labels as a numpy array
    datasetList = list(iter(dataset))
    features = np.vstack(tuple([item[0] for item in datasetList]))
    labels = np.vstack(tuple([item[1] for item in datasetList]))
    return (features,labels)


def getDatasetShape(dataset, featureName = None):
    # Get the shape of the features, number of classes and samples per batch in a dataset
    
    # Single pass to get size of elements in dataset
    for _,batch in enumerate(dataset):
        featureShape = batch[0].shape[-3:] if featureName is None else batch[0][featureName].shape[-3:] # Shape of features
        numClasses = list(batch[1].values())[0].shape if type(batch[1]) is dict else batch[1].shape[-1] # Shape of labels
        batchSize = batch[0].shape[0] if featureName is None else batch[0][featureName].shape[0] # samples per batch if not sequential, patients per batch if it is
        sampsPerTimeSeries = (batch[0].shape[1] if batch[0].ndim == 5 else None) if featureName is None else (batch[0][featureName].shape[1] if batch[0][featureName].ndim == 5 else None)  # Samples per patient, if data is sequential
        break
    return (featureShape,numClasses,batchSize,sampsPerTimeSeries)
    
    
def getSTFT(eeg,fs,downSampleTo = None,epochDur = 30):
    # Helper function for extracting STFT from single channel of EEG
    # 
    # INPUTS
    # eeg -------------------------- (numpy array) eeg to extract data from
    # fs --------------------------- (float or int) Sampling frequency (Hz)
    #                                of eeg data.
    # downSampleTo ----------------- (float or int) Sampling frequency (HzO
    #                                to downsample to prior to data extraction.
    #                                If None, does no downsampling. (Default = None)
    # epochDur --------------------- (float or int) Duration (seconds) of each epoch
    #                                (Default = 30)
    # OUTPUT ----------------------- (4D numpy array) STFT of each epoch. Data is in
    #                                 order [epoch,frequency,time,channel]. 
    
    if (downSampleTo is not None) and (downSampleTo != fs):
        # Downsample signal if necessary
        print('Downsampling to ' + str(downSampleTo))
        eeg = sg.resample_poly(eeg,downSampleTo,fs,axis=1)
        fs = downSampleTo
            
    # Normalize signal from each patient
    eeg = zscore(eeg,axis=1)
    
    # Determine dimensions of stft by taking stft of first epoch and finding dimensions
    stftDims = sg.stft(eeg[0,0:(fs*epochDur)],fs=fs,nperseg=fs,window='hamming',boundary=None)[2].shape
    
    # Initialize dataset
    numEpochs = int(eeg.shape[1]/(fs*epochDur))
    
    dataSet = np.empty((numEpochs,stftDims[0],stftDims[1],eeg.shape[0]),dtype="float16")
    dataSet[:,:,:,:] = np.NaN # Filling with nan aids in error checking later
    
    # Cycle through each epoch and extract data
    for epoch in range(numEpochs):
        for channel in range(eeg.shape[0]):
            # Obtain Features
            _,_,dataSet[epoch,:,:,channel] = np.abs(sg.stft(eeg[channel,(epoch*fs*epochDur):((epoch+1)*fs*epochDur)], \
                                                            fs=fs,nperseg=fs,window='hamming',boundary=None))
        
    # Sanity check - presence of nan values suggests some data was not added to dataSet
    assert not np.isnan(dataSet).any(), 'Unknown Error: dataset contains nan values'
            
    return dataSet

def stftAndCovariance(eeg,fs,downSampleTo = None,epochDur = 30):
    # Helper function for extracting STFT from single channel of EEG
    # 
    # INPUTS
    # eeg -------------------------- (numpy array) eeg to extract data from
    # fs --------------------------- (float or int) Sampling frequency (Hz)
    #                                of eeg data.
    # downSampleTo ----------------- (float or int) Sampling frequency (HzO
    #                                to downsample to prior to data extraction.
    #                                If None, does no downsampling. (Default = None)
    # epochDur --------------------- (float or int) Duration (seconds) of each epoch
    #                                (Default = 30)
    # OUTPUT ----------------------- (4D numpy array) STFT of each epoch. Data is in
    #                                 order [epoch,frequency,time,channel]. 
    
    if (downSampleTo is not None) and (downSampleTo != fs):
        # Downsample signal if necessary
        print('Downsampling to ' + str(downSampleTo))
        eeg = sg.resample_poly(eeg,downSampleTo,fs,axis=1)
        fs = downSampleTo

    # Apply bandpass filter
    sos = sg.butter(5,[.5,40],btype='bandpass',fs=fs,output='sos')
    eeg = sg.sosfilt(sos,eeg)
        
    # Normalize signal from each patient
    eeg = zscore(eeg,axis=1)
    
    # Determine dimensions of stft by taking stft of first epoch and finding dimensions
    stftDims = sg.stft(eeg[0,0:(fs*epochDur)],fs=fs,nperseg=fs,window='hamming',boundary=None)[2].shape
    
    # Initialize dataset
    numEpochs = int(eeg.shape[1]/(fs*epochDur))
    stft = np.empty((numEpochs,stftDims[0],stftDims[1],eeg.shape[0]),dtype="float16")
    stft[:,:,:,:] = np.NaN # Filling with nan aids in error checking later
    covariance = np.empty((numEpochs,eeg.shape[0],eeg.shape[0]))
    
    # Cycle through each epoch and extract data
    for epoch in range(numEpochs):
        for channel in range(eeg.shape[0]):
            # Obtain Features
            _,_,stft[epoch,:,:,channel] = np.abs(sg.stft(eeg[channel,(epoch*fs*epochDur):((epoch+1)*fs*epochDur)], \
                                                            fs=fs,nperseg=fs,window='hamming',boundary=None))
            
        # Calculate Covariance Matrix
        covariance[epoch] = np.cov(eeg[:,(epoch*fs*epochDur):((epoch+1)*fs*epochDur)])
        
    # Sanity check - presence of nan values suggests some data was not added to stft
    assert not np.isnan(stft).any(), 'Unknown Error: dataset contains nan values'
    
    return {'stft':stft,'covariance':covariance}


def getBurgCoefs(eeg,fs,downSampleTo = None,epochDur = 30,order = 8):
    # Extracts average and standard deviation of Burg reflection coefficients
    # over each epoch of eeg data.
    # 
    # INPUTS
    # eeg -------------------------- (numpy array) eeg to extract data from
    # fs --------------------------- (float or int) Sampling frequency (Hz)
    #                                of eeg data.
    # downSampleTo ----------------- (float or int) Sampling frequency (HzO
    #                                to downsample to prior to data extraction.
    #                                If None, does no downsampling. (Default = None)
    # epochDur --------------------- (float or int) Duration (seconds) of each epoch
    #                                (Default = 30)
    # order ------------------------ (int) Order of autoregressive model used in
    #                                Burg's algorithm.
    # OUTPUT ----------------------- (4D numpy array) of extracted data in order
    #                                [epoch,coefficient,{average,standard deviation},1]
    #                                Notice that the last dimension is unused and exists
    #                                only to be compatible with tensorflow expected input
    #                                shape.
    
    # Make sure eeg is 1-D, single-channel array
    if len(eeg.shape) != 1:
        assert min(eeg.shape) == 1, 'Expected single-channel signal for Features extraction, received multi-channel signal'
        eeg = eeg.flatten()
        
    if (downSampleTo is not None) and (downSampleTo != fs):
        # Downsample signal if necessary
        print('Downsampling to ' + str(downSampleTo))
        eeg = sg.resample_poly(eeg,downSampleTo,fs)
        fs = downSampleTo

    # Normalize signal from each patient
    eeg = (eeg - np.mean(eeg))/np.std(eeg)

    # Initialize dataset
    numEpochs = int(np.max(eeg.shape)/(fs*epochDur))
    
    dataSet = np.empty((numEpochs,order,2,1),dtype="float16")
    dataSet[:,:,:,:] = np.NaN # Filling with nan aids in error checking later

    # Cycle through each epoch and extract data
    for epoch in range(numEpochs):
        
        # Obtain reflection coefficients of each second of data in the epoch
        reflectionCoefs = np.zeros((epochDur,order))
        for second in range(epochDur):
            reflectionCoefs[second,:] = arburg_mod(eeg[((epoch*epochDur + second)*fs):((epoch*epochDur + second + 1)*fs)],order)[2]

        # Take average and standard deviation of each coefficient
        dataSet[epoch,:,0,0] = np.nanmean(reflectionCoefs,0)
        dataSet[epoch,:,1,0] = np.nanstd(reflectionCoefs,0)

    dataSet[np.where(np.isnan(dataSet))] = 0 # Handle NaN's by setting them to 0
    return dataSet

def extractRecordsParallel(ExtractionFunc,dirList,*args,ignoreNumPts = False):
    # Helper function for parallelizing Features extraction code
    # 
    # INPUTS
    # ExtractionFunc -------------------- (function) Function to run on each file
    # dirList --------------------------- (string list) List of directories or files to extract from
    # ignorNumPts ----------------------- (bool) If False, will throw an error if the number of
    #                                     of records extracted is different from the number of 
    #                                     records in the original list, which may indicate an error.
    #                                     (Default = False)
    # OUTPUTS --------------------------- List of dictionaris containing the extracted features (numpy
    #                                     array), labels (numpy array) and recording name (string) for
    #                                     each recording.
    
    epochDur = 30 # Epoch duration in seconds
    num_cores = multiprocessing.cpu_count()
    ptList = Parallel(n_jobs=num_cores)(delayed(ExtractionFunc)(record,epochDur,*args) for record in dirList) #[ExtractionFunc(record,epochDur,*args) for record in dirList] 
    
    assert ignoreNumPts or ((None not in ptList) and len(ptList) == len(dirList)), 'Unknown Error: fewer than expected number of patients in dataset'
        
    ptList = [pt for pt in ptList if pt] # Remove empty elements (these occur if there is an error reading the file)    
    return ptList

def extractFromCicc_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromCicc
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Get sampling frequency
    fs = float(os.popen('cd ' + record + ';wfdbdesc ' + record.split("/")[-2] \
                        + ' | grep "Sampling frequency"').read().split()[-2])

    # Get indices of desired channel
    signalInds = [None]*len(possibleChannels)
    for i,channel in enumerate(possibleChannels):
        signalInds[i] = int(os.popen('cd ' + record + ';wfdbdesc ' + record.split("/")[-2] \
                                     + ' | grep "Description: ' + channel + '" -B 2 | head -1').read().split()[-1][0:-1])

    # Extract EEG
    fileName = record + record.split("/")[-2] + '.mat'
    signalList = loadmat(fileName)['val']
    eeg = [signalList[i] for signalInd in signalInds]

    # Extract labels
    fileName = record + record.split("/")[-2] + '-arousal.mat'
    labelDataset = h5py.File(fileName,'r')
    labels = np.array(labelDataset['data']['sleep_stages']['wake'])*stages['awake'] \
             + np.array(labelDataset['data']['sleep_stages']['nonrem1'])*stages['n1'] \
             + np.array(labelDataset['data']['sleep_stages']['nonrem2'])*stages['n2'] \
             + np.array(labelDataset['data']['sleep_stages']['nonrem3'])*stages['n3'] \
             + np.array(labelDataset['data']['sleep_stages']['rem'])*stages['rem'] \
             + np.array(labelDataset['data']['sleep_stages']['undefined'])*stages['undefined']
    
    # Downsample labels so that there is only one label per epoch
    labels = labels[0,0::int(fs*epochDur)]
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    datasetLength = dataSet['stft'].shape[0] if dataSet.__class__ is dict else dataSet.shape[0]

    assert (labels.shape[0] - datasetLength <= 1) and (labels.shape[0] - datasetLength >= 0), \
        'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels[0:datasetLength],'patientId':record.split("/")[-2]}

def extractFromWsc_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromWsc
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]
    
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']

    # Extract labels
    labelFile = record[:-4] + '.eannot'
    file = open(labelFile)
    labels = file.readlines()
    file.close()

    # Translate labels to specified label interpretation
    originalStages = {'N3\n':'n3','NREM3\n':'n3','N2\n':'n2','NREM2\n':'n2','NREM1\n':'n1','N1\n':'n1','REM\n':'rem','wake\n':'awake','?\n':'undefined','\n':'undefined','L\n':'undefined'}
    labels = np.array([stages[originalStages[x]] for x in labels])

    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    datasetLength = dataSet['stft'].shape[0] if dataSet.__class__ is dict else dataSet.shape[0]
    assert labels.shape[0] - datasetLength == 0, 'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels[0:datasetLength],'patientId':record.split("/")[-1]}

def extractFromShhs_parallel(labelDir,record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromShhs
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)
    
    print("Obtaining data from record " + record)
    
    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]
    
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']
    
    # Extract labels
    labelFile = labelDir + record[-16:-4] + '-ann.mat'
    if not path.exists(labelFile):
        print('Label file not found. Skipping entry.')
        return {}
    labels = loadmat(labelFile)['stages'][0]
    
    # Translate labels to specified label interpretation
    if 6 in labels: print('Number 6 found in labels')
    originalStages = {0:'awake',1:'n1',2:'n2',3:'n3',4:'n3',5:'rem',6:'awake',9:'awake'}
    labels = np.array([stages[originalStages[x]] for x in labels])
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels[0:dataSet.shape[0]],'patientId':record.split("/")[-1]}

def extractFromSleepProfiler_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromSleepProfiler
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)

    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]
    
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']

    # Extract labels
    if path.exists(record[:-4] + '.txt'):
        # Code for when label is stored in .txt
        labelFile = record[:-4] + '.txt'
        file = open(labelFile)
        labels = [float(x.split('\t')[1]) for x in file.readlines()[5:-1]]
        file.close()
    elif path.exists(record[:-4] + '.csv'):
        # Code for when label is stored in .csv
        labelFile = record[:-4] + '.csv'
        labels = pd.read_csv(labelFile)["Stage"]
        
    # Translate labels to specified label interpretation
    originalStages = {3.0:'n3',2.0:'n2',1.0:'n1',5.0:'rem',0.0:'awake',21.0:'undefined'}
    labels = np.array([stages[originalStages[x]] for x in labels])
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
        
    return {'features':dataSet,'labels':labels,'patientId':record.split('/')[-1].split('_')[1]}


def extractFromMass_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromMass
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)

    # Find which of several possible channel names is used for the desired EEG electrode location
    channels = [None]*len(possibleChannels)
    headerInfo = edf.read_edf_header(record)
    for i,channel in enumerate(possibleChannels):
        # Find which of several possible channel references used
        if ('EEG ' + channel + '-CLE') in headerInfo['channels']:
            channels[i] = 'EEG ' + channel + '-CLE'
        elif ('EEG ' + channel + '-LER') in headerInfo['channels']:
            channels[i] = 'EEG ' + channel + '-LER'
        else:
            print('Channel not found. Skipping entry.')
            return {}
    
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']
    
    # Extract labels
    labelFile = record[:-7] + 'Base.edf'
    if not path.exists(labelFile):
        print('Label file not found. Skipping entry.')
        return {}
    headerInfo = edf.read_edf_header(labelFile)
    if headerInfo['annotations'][0][1] != b'30':
        print('Wrong epoch size. Skipping entry.')
        return {}
    annotations = [annotation[2] for annotation in headerInfo['annotations']]

    # Because annotations begin and end at different times from the recording, shave off begining and end of EEG signal
    timeDelay = headerInfo['annotations'][0][0]
    timeEnd = headerInfo['annotations'][-1][0] + 30
    sampDelay = int(timeDelay*fs)
    sampEnd = int(timeEnd*fs)+1
    eeg = eeg[:,sampDelay:sampEnd]
    
    # Translate labels to specified label interpretation
    originalStages = {'Sleep stage W':'awake','Sleep stage 1':'n1','Sleep stage 2':'n2','Sleep stage 3':'n3','Sleep stage 4':'n3','Sleep stage R':'rem','Sleep stage ?':'undefined'}
    labels = np.array([stages[originalStages[x]] for x in annotations])
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels[0:dataSet.shape[0]],'patientId':record.split("/")[-1]}

def extractFromIsruc_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromIsruc
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]
    
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']

    # Extract labels
    labelFile = record[:-4] + '_1.txt'
    file = open(labelFile)
    labels = file.readlines()
    if labels[-1] == '\n': labels = labels[:labels.index('\n')] # Remove trailing '\n'
    file.close()

    # Translate labels to specified label interpretation
    originalStages = {'3\n':'n3','2\n':'n2','1\n':'n1','5\n':'rem','0\n':'awake'}
    labels = np.array([stages[originalStages[x]] for x in labels])

    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels[0:dataSet.shape[0]],'patientId':record.split("/")[-1]}
   
def extractFromSSC_SSC_parallel(record,epochDur,downSampleTo,channel,stages,extractFeatures):
    # Helper function to run in parallel for SSC/SSC and SSC/CNC
    #
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Check for EDF/BDF Compliance
    try:
        headerInfo = edf.read_edf_header(record)
    except OSError:
        print('File not EDF(+) or BDF(+) compliant. Skipping entry.')
        return {}

    # Make sure desired channel is present
    if channel not in headerInfo['channels']:
        print('Desired channel not present. Skipping entry.')
        return {}

    # Extract labels
    labelFile = record[:-4] + '.STA'
    if not path.exists(labelFile):
        print('Label file not found. Skipping entry.')
        return {}
    file = open(labelFile)
    labels = [float(x.split('   ')[-1][:-1]) for x in file.readlines()]
    file.close()

    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channel,verbose=False)
    fs = header[0]['sample_rate']
    
    # Translate labels to specified label interpretation (NOTE: not sure if these are correct)
    originalStages = {4.0:'n3',3.0:'n3',2.0:'n2',1.0:'n1',5.0:'rem',0.0:'awake',7.0:'undefined',8.0:'undefined',9.0:'undefined',10.0:'undefined'}
    labels = np.array([stages[originalStages[x]] for x in labels])

    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)

    # Remove ending label if it only covers part of an epoch
    if labels.shape[0] - dataSet.shape[0] == 1: labels = labels[0:-1]

    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
        
    return {'features':dataSet,'labels':labels,'patientId':record[:-4]}


def extractFromSSC_IS_RC_parallel(record,epochDur,downSampleTo,channel,stages,extractFeatures):
    # Helper function to run in parallel for IS-RC
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Check for EDF/BDF Compliance
    try:
        headerInfo = edf.read_edf_header(record)
    except OSError:
        print('File not EDF(+) or BDF(+) compliant. Skipping entry.')
        return {}
        
    # Make sure desired channel is present
    if channel not in headerInfo['channels']:
        print('Desired channel not present. Skipping entry.')
        return {}

    # Extract labels
    labelFile = record[:-4] + '.STA'
    if not path.exists(labelFile):
        print('Label file not found. Skipping entry.')
        return {}
    allLabels = np.genfromtxt(labelFile,delimiter=', ',skip_header=1) # All labels from each rater
    
    # If single mode among scores, label as this mode, otherwise label as undefined
    labels = [y.most_common()[0][0] if (len(y) == 1 or y.most_common()[0][1] > y.most_common()[1][1]) else 7.0
              for y in [Counter(x) for x in allLabels]]
        
    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channel,verbose=False)
    fs = header[0]['sample_rate']
        
    # Translate labels to specified label interpretation (NOTE: not sure if these are correct)
    originalStages = {4.0:'n3',3.0:'n3',2.0:'n2',1.0:'n1',5.0:'rem',0.0:'awake',7.0:'undefined'}
    labels = np.array([stages[originalStages[x]] for x in labels])
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    labels = labels[0:dataSet.shape[0]] # Discard labels past point where EEG recording ends

    # Remove ending label if it only covers part of an epoch
    if labels.shape[0] - dataSet.shape[0] == 1: labels = labels[0:-1]

    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
    
    return {'features':dataSet,'labels':labels,'patientId':record[:-4]}

def extractFromSSC_DHC_parallel(record,epochDur,downSampleTo,channel,stages,extractFeatures):
    # Helper function to run in parallel for SSC/DHC
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    eeg = loadmat(record)[channel] # Extract EEG
    labels = loadmat(record.rsplit('/',1)[0] + '/vec_hypnogram.mat')['hypnogram'][:,0] # Extract labels
        
    # Translate labels to specified label interpretation (NOTE: not sure if these are correct)
    originalStages = {-3:'n3',-2:'n2',-1:'n1',0:'rem',1:'awake',9:'undefined'}
    labels = np.where(np.isnan(labels),9,labels) # Avoids issues in trying to map nan values using a dictionary 
    labels = np.array([stages[originalStages[x]] for x in labels])
    
    # Extract Features
    dataSet = extractFeatures(eeg,256,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Remove ending label if it only covers part of an epoch
    if labels.shape[0] - dataSet.shape[0] == 1: labels = labels[0:-1]
    
    # Sanity check
    assert labels.shape[0] - dataSet.shape[0] == 0, 'Unexpectected mismatch in size between data and labels'
        
    return {'features':dataSet,'labels':labels,'patientId':record[:-4]}
     
def extractFromMrOs_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function to run in parallel for extractFromMros
    # 
    # INPUTS
    # record ------------------------ (string) Record containing data
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    epochDur = 30 # Epoch duration in seconds
    
    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]
    
    # Extract events
    labelPathInfo = record.split('edfs')
    labelFile = labelPathInfo[0] + 'annotations-events-nsrr' + labelPathInfo[1][:-4] + '-nsrr.xml'
    if not path.exists(labelFile):
        print('Label file not found. Skipping entry.')
        return {}
    xmlData = ET.parse(labelFile).getroot()
    eventData = [(event[1].text,float(event[3].text)) for event in xmlData[2][:] if event[0].text == 'Stages|Stages']

    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels ,verbose=False)
    fs = header[0]['sample_rate']
    
    # Re-reference signals to contralateral mastoid
    refEeg,_,_ = edf.read_edf(record,ch_names=['A1','A2'],verbose=False)
    for i,channel in enumerate(channels):
        if (int(channel[-1]) % 2) == 0:
            eeg[i] = eeg[i] - refEeg[0]
        elif (int(channel[-1]) % 2) == 1:
            eeg[i] = eeg[i] - refEeg[1]
    
    # Expand events into epoch-by-epoch labels
    originalStages = {'Stage 4 sleep|4':'n3','Stage 3 sleep|3':'n3','Stage 2 sleep|2':'n2','Stage 1 sleep|1':'n1','REM sleep|5':'rem','Wake|0':'awake','Unscored|9':'undefined'}
    labels = np.array(sum([[stages[originalStages[event[0]]]]*int(event[1]/epochDur) for event in eventData],[]))

    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # If unexpected mismatch in file size, discard excess EEG data but show warning
    if (labels.shape[0] - dataSet.shape[0] != 0):
        print('Warning: unexpected mismatch in file size')
        dataSet = dataSet[0:labels.shape[0]]
    
    return {'features':dataSet,'labels':labels[0:dataSet.shape[0]],'patientId':record.split("/")[-1]}

def extractFromEegBudsAnn_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # Helper function for extracting data from EEGBud .ann and synced .edf files
    # 
    # INPUTS
    # subjectDir -------------------- (string) Directory containing .edf with data and .ann file with sleep stage labels
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)
    
    print("Obtaining data from record " + record)
    subjectName = record.split('/sess_')[0].split('/')[-1]

    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]

    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']
    
    # Define possible labels
    stagesTranslator = {'n3':stages['n3'],'n2':stages['n2'],'n1':stages['n1'],'rem':stages['rem'],'wake':stages['awake'],
                        's3':stages['n3'],'s2':stages['n2'],'s1':stages['n1'],'awake':stages['awake'],'sawake':stages['awake'],'r':stages['rem'],
                        'stage 3':stages['n3'],'stage 2':stages['n2'],'sleep onset (n1)':stages['n1'],'stage 1':stages['n1'],'x':stages['undefined'],
                        '':stages['undefined'],'awake sz':stages['undefined'],'awakie':stages['awake'],'w':stages['awake']}

    labelFile = record[:-4] + '.ann'
    if not path.exists(labelFile):
        # Handle multiple possible naming conventions
        labelFile = record.rsplit('/',1)[0] + '/Aligned_' + subjectName + '_' + record.rsplit('/',1)[1][:-4] + '.ann'
    
    # Extract labels
    labels = get_ann(labelFile, stagesTranslator, int(eeg.shape[-1]/fs))
        
    # If last label takes up less than one full epoch, remove it
    if labels.shape[-1] - int(eeg[:,:int(labels.shape[-1]*epochDur*fs)].shape[-1]/(fs*epochDur)) == 1: labels = labels[:-1]

    # Extract Features
    dataSet = extractFeatures(eeg[:,:int(labels.shape[-1]*epochDur*fs)],fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Sanity check
    assert len(labels) - dataSet.shape[0] == 0, ('Unexpectected mismatch in size between data and labels for record ' + record)
    
    return {'features':dataSet,'labels':np.array(labels),'patientId':subjectName} #,'containsArtifact':containsArtifact}


def get_ann(filename, SleepStages, endTime, debug=False):
    # Extracts labels from .ann file 'filename' using dictionary mapping possible annotation strings 
    # to numerical labels 'SleepStages'. 'endTime' should specify the duration of the EEG recording
    # in seconds.
    # 
    # Modified from code originally written by Hardik Prajapati.
    
    try:
        df = pd.read_csv(filename)
    except:
        return None
        
    stages = df["Annotation"].unique()
    df["AnnotationInt"] = 0
    
    if debug: print(" ==> Freq: {}".format(Counter(df["Annotation"])))

    for stage in stages:
        fil_stage = stage.strip().lower().replace("?", "")
        fil_stage = SleepStages.get(fil_stage, SleepStages["x"])

        if debug: print("   ==> mapping '{}' to '{}'".format(stage, fil_stage))

        idx = df["Annotation"] == stage
        #df.loc[idx, "Annotation"] = fil_stage
        df.loc[idx, "AnnotationInt"] = fil_stage #SleepStageInt[fil_stage]

    start_times = df["Start time"].to_list()
    end_times = df["End time"].to_list()
    annotations = df["AnnotationInt"]

    zipped_index = list(zip(start_times, start_times[1:] + [end_times[-1]], annotations))
    index = list(map(lambda x: np.arange(int(x[0]), int(x[1])), zipped_index))
    index = np.hstack(index)

    value = list(map(lambda x: [x[2]] * (int(x[1]) - int(x[0])), zipped_index))
    value = np.hstack(value)

    annotations_per_sec = np.ones(int(max(index) + 1))*SleepStages["x"]
    
    annotations_per_sec[index] = value

    annotations_per_30_sec = annotations_per_sec[:endTime:30]

    return annotations_per_30_sec.astype(int)


def extractFromEegBudsRml_parallel(record,epochDur,downSampleTo,possibleChannels,stages,extractFeatures):
    # INPUTS
    # record ------------------------ (string) Location of file containing .edf/.bdf and .rml file for a single recording
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channels ---------------------- (list) List of which channels, specified as column number in the .LOG file, to 
    #                                 extract data from.
    # stages ------------------------ (dictionary) A dictionary mapping a sleep stage name (string) to a numerical 
    #                                 label (int)
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array), labels (numpy array)  
    #                                 and recording name (string)

    print("Obtaining data from record " + record)
    
    # Find which of several possible channel names is used for the desired EEG electrode location
    bdfFileList = glob(record + '/*_combined.bdf')
    assert len(bdfFileList) == 1, ("Expected exactly 1 .bdf file in directory " + record)

    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(bdfFileList[0])
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]

    # Extract EEG
    eeg,header,_ = edf.read_edf(bdfFileList[0],ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']

    # Extract sleep events from .rml file
    labelFileList = glob(record + '/*.rml')
    assert len(labelFileList) == 1, ("Expected exactly 1 .rml file in directory " + record)
    xmlData = ET.parse(labelFileList[0]).getroot()
    xmlns = '{http://www.respironics.com/PatientStudy.xsd}'
    sleepXml = xmlData.findall(xmlns + 'ScoringData')[0] \
                                 .findall(xmlns + 'StagingData')[0] \
                                 .findall(xmlns + 'UserStaging')[0] \
                                 .findall(xmlns + 'NeuroAdultAASMStaging')[0] \
                                 .findall(xmlns + 'Stage') # Get all xml related to sleep stages
    
    # Get start time of labels
    labelStartTime = xmlData.findall(xmlns + 'Acquisition')[0] \
                            .findall(xmlns + 'Sessions')[0] \
                            .findall(xmlns + 'Session')[0] \
                            .findall(xmlns + 'RecordingStart')[0].text
    
    # Get duration
    labelDuration = int(xmlData.findall(xmlns + 'Acquisition')[0] \
                        .findall(xmlns + 'Sessions')[0] \
                        .findall(xmlns + 'Session')[0] \
                        .findall(xmlns + 'Duration')[0].text)
    labelDuration = (np.floor(labelDuration/30.0)*30).astype(int) # Trim to last complete epoch
    
    # Gets score of contiguous blocks of time where the score is the same
    originalStages = {'NonREM3':stages['n3'],'NonREM2':stages['n2'],'NonREM1':stages['n1'],'REM':stages['rem'],'Wake':stages['awake'],'NotScored':stages['undefined']} # Map sleep stage names in .rml file to numerical label
    eventNames = [originalStages[event.attrib['Type']] for event in sleepXml] 

    # Get duration of each contiguous block by subtracting start times of contiguous block (recording duration appended to end to get
    # duration of final block
    eventDurations = np.diff([int(event.attrib['Start']) for event in sleepXml], append=labelDuration)
    eventEpochNums = (eventDurations/epochDur).astype(int) # Convert from duration in seconds to duration in epochs
    
    # Expand events into epoch-by-epoch labels
    labels = np.concatenate([[eventNames[i]]*eventEpochNums[i] for i in range(len(eventEpochNums))])
    
    # Shave off beginning of labels so that EEG and label start times match
    startTimeDiff = (headerInfo['startdate'] - parser.parse(labelStartTime)).seconds
    labels = labels[np.ceil((startTimeDiff/epochDur)).astype(int):]

    # Shave off enough time from EEG so that EEG recording starts at first complete epoch
    eegShaveOff = np.ceil(startTimeDiff/epochDur)*epochDur - startTimeDiff 
    eeg = eeg[:,(eegShaveOff*fs).astype(int):]
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    # Truncate either features or labels, whichever is longer, so that both have same duration
    if (labels.shape[0] > dataSet.shape[0]): labels = labels[:dataSet.shape[0]]
    if (labels.shape[0] < dataSet.shape[0]): dataSet = dataSet[:labels.shape[0]] 
            
    return {'features':dataSet,'labels':labels,'patientId':bdfFileList[0].split("/")[-1]}

def extractEegBudUnlabeled_parallel(record,epochDur,downSampleTo,possibleChannels,_,extractFeatures):
    # Helper function for extracting unlabeled data from EEGBud .edf files
    # 
    # INPUTS
    # subjectDir -------------------- (string) Directory containing .edf with data and .ann file with sleep stage labels
    # epochDur ---------------------- (int) Duration of an epoch in seconds
    # downSampleTo ------------------ (float) Frequency (Hz) to downsample eeg data to. If None, does not downsample
    # channel ----------------------- (list) Channel in .edf file to use
    # extractFeatures --------------- (function) Function to use for feature extraction
    # 
    # OUTPUTS ----------------------- Dictionary containing the extracted features (numpy array)and recording name (string)
    
    print("Obtaining data from record " + record)
    subjectName = record.split('/')[-1]

    # Find which of several possible channel names is used for the desired EEG electrode location
    headerInfo = edf.read_edf_header(record)
    channels = [None]*len(possibleChannels)
    for i,channelList in enumerate(possibleChannels):
        channel = [channel for channel in headerInfo['channels'] if channel in channelList] # Intersection between lists of channel names
        if not channel:
            print('None of the channels in ' + str(channelList) + ' were present. Only available channels were ' + str(headerInfo['channels']) + '. Skipping entry.')
            return {}
        elif len(channel) > 1:
            print('Ambiguous channel location: ' + str(channel) + ' were all present. Skipping entry.')
            return {}
        channels[i] = channel[0]

    # Extract EEG
    eeg,header,_ = edf.read_edf(record,ch_names=channels,verbose=False)
    fs = header[0]['sample_rate']
    
    # Extract Features
    dataSet = extractFeatures(eeg,fs,downSampleTo = downSampleTo,epochDur = epochDur)
    
    return {'features':dataSet,'patientId':subjectName}


def extractFromCicc(dataDir,downSampleTo = None,possibleChannels = ["C3-M2"],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from 2018 Computing in Cardiology Challenge Dataset
    dirList = glob(dataDir + '/*/')[:936:13]
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(extractFromCicc_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromWsc(dataDir,downSampleTo = None,possibleChannels = [["C3_M2"]],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from Wisconsin Sleep Cohort
    dirList = glob(dataDir + '/*.edf')[:1080:15]
    #dirList = (dirList[::3] + dirList[1::3])[::63] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(extractFromWsc_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromShhs(dataDir,downSampleTo = None,possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"]],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from Sleep Heart Health Study
    dirList = glob(dataDir + '/edfs/shhs1/*.edf')[:5544:77]
    #dirList = (dirList[::3] + dirList[1::3])[0::347] #[0::1000] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(lambda *args: extractFromShhs_parallel(dataDir + '/labels/',*args),
                                  dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromSleepProfiler(dataDir,downSampleTo = None,possibleChannels = [["EEG3"]],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False,dirList = None):
    # Extracts Features from Sleep Profiler Dataset
    if dirList is None: dirList = glob(dataDir + '/*.edf')
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromSleepProfiler_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromMass(dataDir,downSampleTo = None,possibleChannels = "C3",stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from MASS dataset
    dirList = glob(dataDir + '/*/*PSG.edf')[:144:2]
    #dirList = (dirList[::3] + dirList[1::3])[10::11] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(extractFromMass_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromIsruc(dataDir,downSampleTo = None,possibleChannels = ["C3-A2","C3-M2","C3"],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from ISRUC dataset
    dirList = (glob(dataDir + '/subgroupI/*/*.rec') + glob(dataDir + '/subgroupII/*/*1.rec') + glob(dataDir + '/subgroupIII/*/*1.rec'))[:72]
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(extractFromIsruc_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)
    
def extractFromSSC_SSC(dataDir,downSampleTo,possibleChannels,stages,extractFeatures=getSTFT,debug = False):
    # Helper function for extracting Features from SSC and CNC subdirectories of SSC dataset
    dirList = (glob(dataDir + '/SSC/*.EDF') + glob(dataDir + '/CNC/*.edf'))
    #dirList = dirList[::3] + dirList[1::3] #[0:6] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromSSC_SSC_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromSSC_IS_RC(dataDir,downSampleTo,possibleChannels,stages,extractFeatures=getSTFT,debug = False):
    # Helper function for extracting Features from IS-RC subdirectory of SSC dataset
    # CAUTION: Not sure if labels and EEG signal aligned correctly
    dirList = glob(dataDir + '/IS-RC/*.edf')
    #dirList = dirList[::3] + dirList[1::3] #[0:6] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromSSC_IS_RC_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromSSC_DHC(dataDir,downSampleTo,possibleChannels,stages,extractFeatures=getSTFT,debug = False):
    # Helper function for extracting Features from DHC subdirectory of SSC dataset
    dirList = (glob(dataDir + '/*/*/*/' + possibleChannels + '.mat') + glob(dataDir + '/*/*/' + possibleChannels + '.mat'))
    #dirList = dirList[::3] + dirList[1::3] #[0:6] # Skip every third file for holdout set
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromSSC_DHC_parallel,dirList,downSampleTo,possibleChannels.replace('-',''),stages)
    
def extractFromSSC(dataDir,downSampleTo = None,possibleChannels = 'C3-A2',stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from SSC dataset
    return extractFromSSC_DHC(dataDir,downSampleTo,possibleChannels.lower(),stages,extractFeatures,debug = debug) \
        + extractFromSSC_IS_RC(dataDir,downSampleTo,possibleChannels,stages,extractFeatures,debug = debug) \
        + extractFromSSC_SSC(dataDir,downSampleTo,possibleChannels,stages,extractFeatures,debug = debug)

def extractFromMrOs(dataDir,downSampleTo = None,possibleChannels = "C3",stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from MrOS dataset
    dirList = glob(dataDir + '/polysomnography/edfs/visit1/*.edf')[:2880:40]
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    print(str(len(dirList)) + ' subjects in source dataset.')
    return extractRecordsParallel(extractFromMrOs_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)

def extractFromEegBudsAnn(dataDir,downSampleTo = None,possibleChannelss = [['1']],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from EEGBud Dataset for recordings with data stored in .ann
    dirList = glob(dataDir + '/*/sess_?/*.edf') 
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromEegBudsAnn_parallel,dirList,downSampleTo,possibleChannelss,stages,extractFeatures)

def extractFromEegBudsRml(dataDir,downSampleTo = None,possibleChannelss = [['1']],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from EEGBud Dataset for recordings with labels stored in .rml files
    dirList = glob(dataDir + '/esc_???/data/sess_???/eeg') #[0:6]
    #possibleChannelssAsTuple = tuple([int(possibleChannels) for possibleChannels in possibleChannelss]) # Convert possibleChannels names to integers
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractFromEegBudsRml_parallel,dirList,downSampleTo,possibleChannelss,stages,extractFeatures)

def extractFromEegBuds(dataDir,downSampleTo = None,possibleChannels = [['1']],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from EEGBud Dataset
    return extractFromEegBudsAnn(dataDir,downSampleTo,possibleChannels,stages,extractFeatures) \
        + extractFromEegBudsRml(dataDir,downSampleTo,possibleChannels,stages,extractFeatures) 

def extractEegBudUnlabeled(dataDir,downSampleTo = None,possibleChannels = [['1']],stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5},extractFeatures=getSTFT,debug = False):
    # Extracts Features from EEGBud Dataset for recordings with no labels
    dirList = glob(dataDir + '/unlabeled/*.edf') + glob(dataDir + '/unlabeled/*.bdf')
    if debug: dirList = dirList[0:4] # Only use a few subjects during debugging
    return extractRecordsParallel(extractEegBudUnlabeled_parallel,dirList,downSampleTo,possibleChannels,stages,extractFeatures)
