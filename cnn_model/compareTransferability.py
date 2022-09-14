#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.backend import function
import os
import itertools
os.environ['PYTHONHASHSEED']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses certain tensorflow output
import utils
import numpy as np
import itertools as itr
import tensorflow as tf
from scipy.stats import mode, entropy
from scipy.spatial.distance import pdist, squareform
from models import CNN_2D, CoralTransfer, DeepDomainConfusion, HeadRetrain, SubspaceAlignment
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from modCoral import RegCORAL, SCORAL
from transferMeasures import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, silhouette_score
from sleep_staging_main import validationListAssembler
from xlwt import Workbook

# Define inputs
architecture = CNN_2D
srcDataSet = 'mass'
targDataSet = 'SleepProfiler'
savedMdlDir = './massCnn/model'
saveOutputTo = './transferabilityResults_mass'
numFolds = 24
retrainLayers = [-4]
debug = False

numChannels = 1
downSampleTo = 100
extractFeatures = utils.getSTFT
stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5}
epochs = 3 if debug else 1000
d = 10 if debug else 100 # Dimensionality used in subspace alignment

measures = {"Hypothesis Margin":lambda srcAndTarg,srcVsTargLabels,params: hypothesis_margin(srcAndTarg,srcVsTargLabels),
            "Silhouette Score":lambda srcAndTarg,srcVsTargLabels,params: silhouette_score(srcAndTarg,srcVsTargLabels),
            "MMD 1xGamma": lambda srcAndTarg,srcVsTargLabels,params: MMD(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],params['gamma']),
            "MMD 10xGamma": lambda srcAndTarg,srcVsTargLabels,params: MMD(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],10*params['gamma']),
            "MMD .1xGamma": lambda srcAndTarg,srcVsTargLabels,params: MMD(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],.1*params['gamma']),
            "TDAS 1xEpsilon": lambda srcAndTarg,srcVsTargLabels,params: TDAS(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],d,params['epsilon']),
            "TDAS 10xEpsilon": lambda srcAndTarg,srcVsTargLabels,params: TDAS(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],d,10*params['epsilon']),
            "TDAS .1xEpsilon": lambda srcAndTarg,srcVsTargLabels,params: TDAS(srcAndTarg[srcVsTargLabels == 0],srcAndTarg[srcVsTargLabels == 1],d,.1*params['epsilon'])}
            

# Initialize excel sheet
wb = Workbook()
sheets = {layer:wb.add_sheet("Layer " + str(layer),cell_overwrite_ok=True) for layer in retrainLayers}

# Extract source data
if srcDataSet == 'cicc':
    dataDir = '/labs/cliffordlab/data/Challenge2018/training'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromCicc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ['C3-M2','C4-M1'][:numChannels]
    
elif srcDataSet == 'wsc':
    dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromWsc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
    
elif srcDataSet == 'shhs':
    dataDir = '/labs/cliffordlab/data/shhs'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromShhs(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
    
elif srcDataSet == 'SleepProfiler':
    dataDir = '/labs/cliffordlab/data/EEG/Sleep_Profiler'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSleepProfiler(*args,**kwargs,extractFeatures=extractFeatures,dirList = dirList)
    possibleChannels = [['EEG3'],['EEG2']][:numChannels] #range(0,14,3)

elif srcDataSet == 'mass':
    dataDir = '/labs/cliffordlab/data/MASS'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMass(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ['C3','C4'][:numChannels]

elif srcDataSet == 'isruc':
    dataDir = '/labs/cliffordlab/data/ISRUC'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromIsruc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]

elif srcDataSet == 'ssc':
    dataDir = '/labs/cliffordlab/data/SSC'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSSC(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ["C3-A2"][:numChannels]

elif srcDataSet == 'mros':
    dataDir = '/labs/cliffordlab/data/MrOS'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMrOs(*args,**kwargs,extractFeatures=extractFeatures)
    # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
    # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
    possibleChannels = [["C3"],["C4"]][:numChannels]

elif srcDataSet == 'eegbud':
    dataDir = '/labs/cliffordlab/data/EEG/hearables/Sleep_Data'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromEegBuds(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["5","CH5","Ch5"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]

else:
    print('Unknown dataset')

srcData = featureExtractionFunc(dataDir,downSampleTo = downSampleTo, possibleChannels = possibleChannels, stages = stages, debug = debug)


# Extract target data
if targDataSet == 'cicc':
    dataDir = '/labs/cliffordlab/data/Challenge2018/training'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromCicc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ['C3-M2','C4-M1'][:numChannels]
    numFolds = 5
    
elif targDataSet == 'wsc':
    dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromWsc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
    numFolds = 5
    
elif targDataSet == 'shhs':
    dataDir = '/labs/cliffordlab/data/shhs'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromShhs(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
    numFolds = 5
    
elif targDataSet == 'SleepProfiler':
    dataDir = '/labs/cliffordlab/data/EEG/Sleep_Profiler'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSleepProfiler(*args,**kwargs,extractFeatures=extractFeatures,dirList = None)
    possibleChannels = [['EEG3'],['EEG2']][:numChannels] #range(0,14,3)
    numFolds = 24 #14 - len(holdOut)

elif targDataSet == 'mass':
    dataDir = '/labs/cliffordlab/data/MASS'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMass(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ['C3','C4'][:numChannels]
    numFolds = 5

elif targDataSet == 'isruc':
    dataDir = '/labs/cliffordlab/data/ISRUC'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromIsruc(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]
    numFolds = 5

elif targDataSet == 'ssc':
    dataDir = '/labs/cliffordlab/data/SSC'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSSC(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = ["C3-A2"][:numChannels]
    numFolds = 5

elif targDataSet == 'mros':
    dataDir = '/labs/cliffordlab/data/MrOS'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMrOs(*args,**kwargs,extractFeatures=extractFeatures)
    # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
    # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
    possibleChannels = [["C3"],["C4"]][:numChannels]
    numFolds = 5

elif targDataSet == 'eegbud':
    dataDir = '/labs/cliffordlab/data/EEG/hearables/Sleep_Data'
    featureExtractionFunc = lambda *args,**kwargs: utils.extractFromEegBuds(*args,**kwargs,extractFeatures=extractFeatures)
    possibleChannels = [["5","CH5","Ch5"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]
    numFolds = 5

else:
    print('Unknown dataset')

targData = featureExtractionFunc(dataDir,downSampleTo = downSampleTo, possibleChannels = possibleChannels, stages = stages, debug = debug)


# Wrap data in tensorflow dataset objects
for fold in range(numFolds):
    validationList = validationListAssembler(True,fold,srcData,numFolds = numFolds,targetData = targData)
    sourceDataset,targetDataset,earlyStoppingDataset,validationDataset,class_weights = utils.transferDatasetAssembler(srcData,targData,validationList,stages)
    ptIdList = set([targData[pt]['patientId'] for pt in validationList])
    
    # Load pre-trained model
    pretrained = architecture()
    pretrained(np.zeros((1,) + utils.getDatasetShape(sourceDataset)[0])) # Forces model to be built so that weights can be loaded
    pretrained.load_weights(savedMdlDir)
    
    # Construct Set of Models
    transferMethods = [{'name':'Head Retrain','callable':lambda pretrained,retrainLayer: HeadRetrain(pretrained,layerSignifier = retrainLayer)},
                       {'name':'CORAL','callable':lambda pretrained,retrainLayer: CoralTransfer(pretrained,transform = RegCORAL(),layerSignifier = retrainLayer)},
                       {'name':'SCORAL','callable':lambda pretrained,retrainLayer: CoralTransfer(pretrained,transform = SCORAL(),layerSignifier = retrainLayer)},
                       {'name':'DDC','callable':lambda pretrained,retrainLayer: DeepDomainConfusion(pretrained,[retrainLayer])},
                       {'name':'SA','callable':lambda pretrained,retrainLayer: SubspaceAlignment(pretrained,layerSignifier = retrainLayer,n_components=d)}]

    rows = {'Subject':0,'LEEP':1,'H-score':2}
    rowNum = 3
    for iTransferMethod in transferMethods:
        for measureName in measures.keys():
            rows[iTransferMethod['name'] + '-' + measureName + ' Before Re-train'] = rowNum
            rows[iTransferMethod['name'] + '-' + measureName + ' After Re-train'] = rowNum + 1
            rowNum = rowNum + 2

        rows[iTransferMethod['name'] + ' Accuracy'] = rowNum
        rows[iTransferMethod['name'] + ' Precision'] = rowNum + 1
        rows[iTransferMethod['name'] + ' Recall'] = rowNum + 2
        rows[iTransferMethod['name'] + ' F1'] = rowNum + 3
        rows[iTransferMethod['name'] + ' Kappa'] = rowNum + 4
        rowNum = rowNum + 5
    
    # Cycle through each layer to test
    for iRetrainLayer in retrainLayers:
        print('Using Layer ' + str(iRetrainLayer) + ' at fold ' + str(fold))        

        # Print rownames to excel sheet
        for rowName,rowNumber in rows.items():
            sheets[iRetrainLayer].write(rowNumber,0,rowName)
        sheets[iRetrainLayer].write(rows['Subject'],fold+1,str(ptIdList))
        
        # Extract target features
        originalFeatureExtractor,originalClassifer = splitModel(pretrained,layerSignifier = iRetrainLayer)
        originalTargFeaturesDataset = targetDataset.map(lambda x,y: (tf.py_function(originalFeatureExtractor,[x],tf.float32),y),tf.data.experimental.AUTOTUNE)
        originalTargFeatures = np.vstack(tuple([sample[0] for sample in iter(originalTargFeaturesDataset)])) # Convert to numpy
        y_targ = np.argmax(np.vstack(tuple([sample[1] for sample in iter(originalTargFeaturesDataset)])),axis=-1).flatten()
        
        # Extract original and transferred source features
        originalSrcFeaturesDataset = sourceDataset.map(lambda x,y: (tf.py_function(originalFeatureExtractor,[x],tf.float32),y),tf.data.experimental.AUTOTUNE) #.unbatch()
        originalSrcFeatures = np.vstack(tuple([sample[0] for sample in iter(originalSrcFeaturesDataset)])) # Convert to numpy
        y_src = np.argmax(np.vstack(tuple([sample[1] for sample in iter(originalSrcFeaturesDataset)])),axis=-1).flatten()
        
        # Evaluate transferability of the model
        sheets[iRetrainLayer].write(rows['LEEP'],fold+1,str(LEEP(originalTargFeatures,y_targ,originalClassifer)))
        sheets[iRetrainLayer].write(rows['H-score'],fold+1,str(h_score(originalTargFeatures,y_targ)))
        
        # Cycle through each transfer method to use
        for transferMethod in transferMethods:
            print('Using transfer method ' + transferMethod['name'] + ' at layer ' + str(iRetrainLayer) + ' at fold ' + str(fold))
            
            # Make a copy of the pre-trained model
            model = architecture()
            model.build((1,) + utils.getDatasetShape(targetDataset)[0])
            model.set_weights(pretrained.get_weights())
            model = transferMethod['callable'](model,iRetrainLayer) # Construct transfer model from pre-trained model
            
            # Train model
            monitor = 'val_loss' if model.__class__ is DeepDomainConfusion else 'val_accuracy'
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            model.fit(sourceDataset,targetDataset,validation_data=earlyStoppingDataset,epochs=epochs,verbose = 0,shuffle=True,callbacks=callbacks)
            
            # Extract post-retrain source features
            transferredSrcExtractor,transferredTargExtractor,_ = splitModel(model,layerSignifier = iRetrainLayer)
            sourceFeatures,sourceLabels = utils.dataset2Numpy(sourceDataset)
            transferredSrcFeatures = np.vstack(tuple([transferredSrcExtractor(sourceFeatures[None,i],np.argmax(sourceLabels[i],axis=-1)) for i in range(sourceLabels.shape[0])]))
            
            # Extract post-retrain target features
            targetFeatures,targetLabels = utils.dataset2Numpy(targetDataset)
            transferredTargFeatures = np.vstack(tuple([transferredTargExtractor(targetFeatures[None,i],np.argmax(targetLabels[i],axis=-1)) for i in range(targetLabels.shape[0])]))

            # Calculate gamma paramter to use in kernel for MMD calculation
            params = {}
            medianDistBtwSamps = np.median(pdist(originalTargFeatures)) 
            params['gamma'] = -1/(2*medianDistBtwSamps)

            # Set epsilon to the median distance between samples in the untransformed target dataset
            params['epsilon'] = medianDistBtwSamps
            
            # Compare source & target features before and after domain transfer
            srcAndTarg = np.vstack((originalSrcFeatures,originalTargFeatures))
            transformedSrcAndTarg = np.vstack((transferredSrcFeatures,transferredTargFeatures))
            srcVsTargLabels = np.concatenate([np.zeros(originalSrcFeatures.shape[0]),np.ones(originalTargFeatures.shape[0])]) # Mark which samples are from source or target
            combinedSleepStages = np.concatenate([y_src,y_targ])

            # Take each transferability measure before and after domain adapatation
            for measureName,measureFunction in measures.items():
                sheets[iRetrainLayer].write(rows[transferMethod['name'] + '-' + measureName + ' Before Re-train'],fold+1,
                                            str(measureFunction(srcAndTarg,srcVsTargLabels,params)))
                sheets[iRetrainLayer].write(rows[transferMethod['name'] + '-' + measureName + ' After Re-train'],fold+1,
                                            str(measureFunction(transformedSrcAndTarg,srcVsTargLabels,params)))
            
            # Initialize arrays to store classifications and ground truths
            numBatches = utils.getNumBatches(validationDataset)
            _,_,batchSize,sampsPerTimeSeries = utils.getDatasetShape(validationDataset,None)
            totalSamples = numBatches*batchSize*(sampsPerTimeSeries if sampsPerTimeSeries is not None else 1) # Calculate total samples in dataset
            labels = np.empty(totalSamples)
            classifications = np.empty(totalSamples)
            labels[:] = np.nan # Fill with NaN for easy error checking
            classifications[:] = np.nan
            
            predict = (lambda model,x: np.argmax(model.predict(x)["classifier"],axis=-1).flatten()) if model.__class__ is DeepDomainConfusion                     else (lambda model,x: np.argmax(model.predict(x),axis=-1).flatten()) # Get classifications in slightly different way if model has multiple outputs
            
            samplesAdded = 0 # Track number of samples which have been added thus far
            for x,y,*_ in validationDataset:
                # Run through each batch in validation set
                batchClassifications = predict(model,x) # Classify all samples in batch
                classifications[samplesAdded:(samplesAdded + batchClassifications.shape[0])] = batchClassifications
                labels[samplesAdded:(samplesAdded + batchClassifications.shape[0])] = np.argmax(y,axis=-1).flatten()
                samplesAdded = samplesAdded + batchClassifications.shape[0]
                
            # Remove unused space
            classifications = classifications[0:samplesAdded]
            labels = labels[0:samplesAdded]
            
            # Remove unlabeled samples
            classifications = np.delete(classifications,np.where(labels == stages['undefined']))
            labels = np.delete(labels,np.where(labels == stages['undefined']))
            
            # Error check
            assert not np.any(np.isnan(classifications)), "NaN values found in classifications during validation"
            assert not np.any(np.isnan(labels)), "NaN values found in labels during validation"
            
            # Dispaly Confusion Matrix
            print('Confusion Matrix')
            np.set_printoptions(suppress=True)
            print(100*confusion_matrix(classifications,labels,normalize='true').T)
            np.set_printoptions(suppress=False)
            
            # Calculate Performance Metrics
            acc = accuracy_score(labels,classifications)
            prec = precision_score(labels,classifications,average='macro')
            rec = recall_score(labels,classifications,average='macro')
            f1 = f1_score(labels,classifications,average='macro')
            kappa = cohen_kappa_score(labels,classifications)
            
            # Display performance
            print('Accuracy: ' + str(acc))
            sheets[iRetrainLayer].write(rows[transferMethod['name'] + ' Accuracy'],fold+1,str(acc))
            print('Precision: ' + str(prec))
            sheets[iRetrainLayer].write(rows[transferMethod['name'] + ' Precision'],fold+1,str(prec))
            print('Recall: ' + str(rec))
            sheets[iRetrainLayer].write(rows[transferMethod['name'] + ' Recall'],fold+1,str(rec))
            print('F1: ' + str(f1))
            sheets[iRetrainLayer].write(rows[transferMethod['name'] + ' F1'],fold+1,str(f1))
            print('Kappa: ' + str(kappa))
            sheets[iRetrainLayer].write(rows[transferMethod['name'] + ' Kappa'],fold+1,str(kappa))

            # Save Results
            wb.save(saveOutputTo + ('_debug.xls' if debug else '.xls'))
