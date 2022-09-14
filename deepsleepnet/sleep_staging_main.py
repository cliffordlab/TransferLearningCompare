import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import keras_tuner as kt
import os
os.environ['PYTHONHASHSEED']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses certain tensorflow output
import tensorflow as tf
import utils
from models import CNN_2D, CoralTransfer, DeepDomainConfusion, HeadRetrain, SubspaceAlignment
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import sys
import datetime
import time
import faulthandler; faulthandler.enable()

def foldInputParsert(**kwargs):
    # Helper function for parsing inputs to determine number of folds (if any)
    if ('numFolds' in kwargs) and ('validationSet' in kwargs):
        # Error check
        raise Exception('Error: can either use k-fold cross-validation or validate on specified patient set, but not both')
        
    elif 'numFolds' in kwargs:
        # Perform k-fold cross-validation
        kFoldCrossval = True
        numFolds = kwargs['numFolds']
        print('Performing ' + str(numFolds) + '-fold cross-validation')
        
    elif 'validationSet' in kwargs:
        # Test on specific set of patients
        kFoldCrossval = False
        numFolds = 1
        print('Validating on specified patient set')
        
    else:
        # Default to 5-fold cross-validation
        numFolds = 5
        kFoldCrossval = True
        print('Performing ' + str(numFolds) + '-fold cross-validation')

    return (kFoldCrossval,numFolds)

def validationListAssembler(kFoldCrossval,fold,sourceData,numFolds = 5,targetData = None,**kwargs):
    # Helper function for parsing settings and code state to determine which patients will be used
    # in validation
    
    # Get list of all unique subjects in list of recordings
    getUniquePtNames = lambda recordList: np.unique(np.array([record['patientId'] for record in recordList]))
    ptList = getUniquePtNames(sourceData) if targetData is None else getUniquePtNames(targetData)
    
    # Create list of subjects to use for validation
    if kFoldCrossval:
        # If performing k-fold cross-validation, validate on different fold each time
        validationPtList = ptList[range(fold,len(ptList),numFolds)]
    else:
        # Otherwise, just use specified validation set
        validationPtList = ptList[kwargs['validationSet']]

    # Get indices of each record for validation subjects
    getValidationPtIndices = lambda recordList, ptNames: [i for i in range(len(recordList)) if recordList[i]['patientId'] in ptNames]
    validationList = getValidationPtIndices(sourceData,validationPtList) if targetData is None else getValidationPtIndices(targetData,validationPtList)

    return validationList

def main(epochs = 1000,verbose = 0,**kwargs):

    # Print all parameters
    print('Dumping parameter values:')
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
        
    # Ensure reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)

    # GPU Setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Determine type of training
    transfer = ('transfer' in kwargs) and kwargs['transfer']
    adversarialInf = ('adversarialInf' in kwargs) and kwargs['adversarialInf']
    unsupervisedPretrain = ('unsupervisedPretrain' in kwargs) and (kwargs['unsupervisedPretrain'] is not None)
    multiInput = ('multiInput' in kwargs) and kwargs['multiInput']
    ensemble = ('ensemble' in kwargs) and kwargs['ensemble']
    assert (transfer + adversarialInf + unsupervisedPretrain + multiInput + ensemble) <= 1, "Please set either transfer, adversarialInf, unsupervisedPretrain, multiInput, or ensemble options to True, but not more than 1"
    
    # Load Data
    if ('loadDataFrom' in kwargs) and (kwargs["loadDataFrom"] is not None):
        # Load saved features which have already been extracted
        data = np.load(kwargs["loadDataFrom"],allow_pickle=True)
        if (transfer or adversarialInf or ensemble): targetData = np.load(kwargs["loadTransferDataFrom"],allow_pickle=True) # Load target dataset if doing transfer learning
        
    else:
        # Call function for extracting features from raw data
        data = kwargs["dataLoadFunc"](kwargs["dataDir"])
        if (transfer or adversarialInf or ensemble): targetData = kwargs["targetDataLoadFunc"](kwargs["targetDataDir"]) # Load target dataset if doing transfer learning
        if (unsupervisedPretrain and "unlabeledDataLoadFunc" in kwargs and kwargs["unlabeledDataLoadFunc"] is not None):
            unlabeledData = kwargs["unlabeledDataLoadFunc"](kwargs["unlabeledDataDir"])
        else:
            unlabeledData = None
        if ('saveDataTo' in kwargs) and (kwargs['saveDataTo'] is not None):
            # Save extracted features, if specified to do so
            np.save(kwargs['saveDataTo'],data)
        
    # Remove held-out patients, if any
    if 'holdOut' in kwargs:
        # Hold out specified patients for testing by removing them from set
        data = np.delete(data,kwargs['holdOut'])
        print(str(len(kwargs['holdOut'])) + ' patients held out')
        
    # Also remove held-out patients from target dataset, if any
    if (transfer or adversarialInf or ensemble) and ('transferHoldOut' in kwargs):
        targetData = np.delete(targetData,kwargs['transferHoldOut'])
        print(str(len(kwargs['holdOut'])) + ' patients held out from holdout set')
        
    # Either perform k-fold cross-validation on remaining patients or evaluate on specified set
    kFoldCrossval,numFolds = foldInputParsert(**kwargs)
    
    # Perform training/testing on specified subset(s)
    scores = np.empty((0,5)) # Track score on each fold
    for fold in range(numFolds):
        print('Fold ' + str(fold))

        # Assemble datasets
        if transfer or ensemble:
            validationList = validationListAssembler(kFoldCrossval,fold,data,targetData = targetData,**kwargs)
            sourceDataset,targetDataset,earlyStoppingDataset,validationDataset,class_weights = utils.transferDatasetAssembler(data,targetData,validationList,stages,**kwargs)
            
        elif adversarialInf:
            validationList = validationListAssembler(kFoldCrossval,fold,data,targetData = targetData,**kwargs)
            trainDataset,earlyStoppingDataset,validationDataset,class_weights = utils.advInfDatasetAssembler(data,targetData,validationList,stages,**kwargs)
            
        elif unsupervisedPretrain:
            validationList = validationListAssembler(kFoldCrossval,fold,data,**kwargs)
            trainDataset,earlyStoppingDataset,validationDataset,class_weights = utils.datasetAssembler(data,validationList,stages,**kwargs)
            unsupTrainDataset,unsupEarlyStoppingDataset = utils.unsupDatasetAssembler(data,validationList,stages,unlabeledData = unlabeledData,**kwargs)
            
        elif multiInput:
            validationList = validationListAssembler(kFoldCrossval,fold,data,**kwargs)
            trainDataset,earlyStoppingDataset,validationDataset,class_weights = utils.datasetMultiInputAssembler(data,validationList,stages,**kwargs)
            
        else:
            validationList = validationListAssembler(kFoldCrossval,fold,data,**kwargs)
            trainDataset,earlyStoppingDataset,validationDataset,class_weights = utils.datasetAssembler(data,validationList,stages,**kwargs)
            
            
        if kwargs['sequential']:
            featureShape,_,_,sampsPerTimeSeries = utils.getDatasetShape(sourceDataset) if (transfer or ensemble) else utils.getDatasetShape(trainDataset) 
            datasetShape = (None,) + (sampsPerTimeSeries,) + featureShape
        else:
            if multiInput:
                stftShape = utils.getDatasetShape(trainDataset,'stft')[0]
                covShape = utils.getDatasetShape(trainDataset,'covariance')[0]
            else:
                datasetShape = (None,) + (utils.getDatasetShape(sourceDataset)[0] if (transfer or ensemble) else utils.getDatasetShape(trainDataset)[0] )
                
        if ('loadMdl' in kwargs) and (kwargs['loadMdl'] is not None):
            # Load saved model
            print('Loading saved model...')
            if multiInput:
                model = kwargs['modelType'](stftShape = stftShape,covShape = covShape)
                model({'stft':np.zeros(stftShape),'covariance':np.zeros(covShape)}) # Forces model to be built so that weights can be loaded
            else:
                model = kwargs['modelType']()
                model.build(datasetShape) # Forces model to be built so that weights can be loaded
            model.load_weights(kwargs['loadMdl'])

        elif unsupervisedPretrain:
            # Pre-train model on unsupervised task before training on main task
            print('Pre-training model on unsupervized task before training on main task')

            # Load the model
            if multiInput:
                model = kwargs['modelType'](stftShape = stftShape,covShape = covShape)
                model({'stft':np.zeros(stftShape),'covariance':np.zeros(covShape)}) # Forces model to be built so that weights can be loaded
            else:
                model = kwargs['modelType']()
                model(np.zeros(datasetShape)) # Forces model to be built so that weights can be loaded
            model.load_weights(kwargs['transferMdl'])
            
            trainUnsupervized(model,kwargs['unsupervisedPretrain'],unsupTrainDataset,epochs,earlyStoppingData = unsupEarlyStoppingDataset,
                              outputShape = utils.getDatasetShape(unsupTrainDataset)[0],tapLayer = -2,freezeBelow = -6)
            
            # Define callbacks
            monitor = 'val_classifier_accuracy' if adversarialInf else 'val_accuracy'
            callbacks = [AdvInfEarlyStopping(patience=30, min_delta = .001)] if adversarialInf else [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)), histogram_freq=1)) # Use tensorboard, if specified
            
            # Freeze every layer except final dense layer
            for layer in model.layers[:-5]: layer.trainable = False
            
            # Train model on main task
            model.fit(trainDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks)
            if 'saveMdl' in kwargs: model.save_weights(kwargs['saveMdl'])

        elif ('retrain' in kwargs) and (kwargs['retrain'] is not None):
            # Re-train an existing model
            print('Re-training existing model')
            
            # Load the model
            if multiInput:
                model = kwargs['modelType'](stftShape = stftShape,covShape = covShape)
                model({'stft':np.zeros(stftShape),'covariance':np.zeros(covShape)}) # Forces model to be built so that weights can be loaded
            else:
                model = kwargs['modelType']()
                model(np.zeros(datasetShape)) # Forces model to be built so that weights can be loaded
            model.load_weights(kwargs['retrain'])
            
            # Freeze every layer except final dense layer
            for layer in model.layers[:-1]: layer.trainable = False
            
            # Define callbacks
            monitor = 'val_classifier_accuracy' if adversarialInf else 'val_accuracy'
            callbacks = [AdvInfEarlyStopping(patience=30, min_delta = .001)] if adversarialInf else [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)), histogram_freq=1)) # Use tensorboard, if specified
            
            # Train model
            model.fit(trainDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks) # class_weight = class_weights
            if 'saveMdl' in kwargs: model.save_weights(kwargs['saveMdl'])
            
        elif transfer:
            # Performing transfer learning on a pre-trained model
            print('Performing transfer learning')
            
            # Load the model
            pretrained = kwargs['modelType']()
            pretrained.build((None,) + utils.getDatasetShape(sourceDataset)[0])
            if ('transferMdl' in kwargs and kwargs['transferMdl'] is not None): pretrained.load_weights(kwargs['transferMdl'])
            model = SynthSampleRetrainer(pretrained,WGAN_GP(),50000,saveTo=kwargs['saveGeneratorTo']) #DeepDomainConfusion(pretrained,['conv4']) #TcaTransfer(pretrained,layerSignifier = -4,n_components=1200,mu=1,sigma=1,geo_sigma2=.0001,gamma=1.0,lambda_=.5,knn=10) #HeadRetrain(pretrained,layerSignifier = -4) #CoralTransfer(pretrained,layerSignifier = -4) #SubspaceAlignment(pretrained,layerSignifier = -4,n_components=800) #WeightedTrainModel(pretrained)
            
            # Define callbacks
            monitor = 'val_loss' if model.__class__ is DeepDomainConfusion else 'val_accuracy'
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)))) # Use tensorboard, if specified
            
            # Perform Transfer Learning From Source to Target Dataset
            model.fit(sourceDataset,targetDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks) # class_weight = class_weights
            
        elif ensemble:
            # Performing transfer learning on a pre-trained model
            print('Performing transfer learning')
            
            # Initialize the model
            transferMethod = lambda pretrained: CoralTransfer(pretrained,layerSignifier = -4)
            model = WeightedSoftVote((None,) + utils.getDatasetShape(sourceDataset)[0],
                                     kwargs['ensembleList'],
                                     kwargs['modelType'],
                                     transferMethod = transferMethod)
            
            # Define callbacks
            monitor = 'val_loss' if model.__class__ is DeepDomainConfusion else 'val_accuracy'
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)))) # Use tensorboard, if specified
            
            # Perform Transfer Learning From Source to Target Dataset
            model.fit(sourceDataset,targetDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks) # class_weight = class_weights
            
        elif ('tune' in kwargs) and kwargs['tune']:
            # Tune model hyperparameters
            print('Tuning model hyperparameters...')
            
            # Define callbacks
            monitor = 'val_classifier_accuracy' if adversarialInf else 'val_accuracy'
            callbacks = [AdvInfEarlyStopping(patience=30, min_delta = .001)] if adversarialInf else [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)), histogram_freq=1)) # Use tensorboard, if specified
            
            # Define tuner
            modelInitializer = lambda x: kwargs['modelType'](x,stftShape = stftShape,covShape = covShape) if multiInput else kwargs['modelType'](x,inputShape = datasetShape[1:])
            tuner = kt.BayesianOptimization(modelInitializer,objective=kt.Objective(monitor,direction="max"),directory=kwargs['logDir'],project_name=str(fold),
                                            overwrite=False,max_trials=kwargs['max_trials'],executions_per_trial=1)
            
            # Train and tune
            tuner.search(trainDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks)
            model = tuner.get_best_models(num_models=1)[0]
            
            # Print results
            print('Tuned Hyperparameter values:')
            for key, value in tuner.get_best_hyperparameters()[0].values.items():
                print("{0} = {1}".format(key, value))
            
        else:
            # Train new model
            print('Training new model...')
            model = kwargs['modelType'](stftShape = stftShape,covShape = covShape) if multiInput else kwargs['modelType'](inputShape = datasetShape[1:])
            
            # Define callbacks
            monitor = 'val_classifier_accuracy' if adversarialInf else 'val_accuracy'
            callbacks = [AdvInfEarlyStopping(patience=30, min_delta = .001)] if adversarialInf else [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
            if 'logDir' in kwargs: callbacks.append(TensorBoard(log_dir=(kwargs['logDir'] + str(fold)), histogram_freq=1)) # Use tensorboard, if specified
            
            # Train model
            model.fit(trainDataset,validation_data=earlyStoppingDataset,
                      epochs=epochs,verbose=verbose,shuffle=True,callbacks=callbacks)
            if 'saveMdl' in kwargs: model.save_weights(kwargs['saveMdl'])

        if not validationList:
            print('No validation patients. Exiting script without testing model performance.')
            return
        
        # Test trained model
        print('Validation Set Performance on fold ' + str(fold) + ':')

        
        # Initialize arrays to store classifications and ground truths
        numBatches = utils.getNumBatches(validationDataset)
        _,_,batchSize,sampsPerTimeSeries = utils.getDatasetShape(validationDataset,'stft' if multiInput else None)
        totalSamples = numBatches*batchSize*(sampsPerTimeSeries if sampsPerTimeSeries is not None else 1) # Calculate total samples in dataset
        labels = np.empty(totalSamples)
        classifications = np.empty(totalSamples)
        labels[:] = np.nan # Fill with NaN for easy error checking
        classifications[:] = np.nan
        
        # Obtain predictions of trained network and corresponding ground truth labels
        predict = (lambda model,x: np.argmax(model.predict(x)["classifier"],axis=-1).flatten()) if (adversarialInf or (model.__class__ is DeepDomainConfusion)) \
            else (lambda model,x: np.argmax(model.predict(x),axis=-1).flatten()) # Get classifications in slightly different way if model has multiple outputs
        
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
        np.set_printoptions(suppress=True)
        print(100*confusion_matrix(classifications,labels,normalize='true').T)
        np.set_printoptions(suppress=False)
        
        # Calculate Performance Metrics
        acc = accuracy_score(classifications,labels)
        prec = precision_score(classifications,labels,average='macro')
        rec = recall_score(classifications,labels,average='macro')
        f1 = f1_score(classifications,labels,average='macro')
        kappa = cohen_kappa_score(classifications,labels)
        
        # Display performance
        print('Accuracy: ' + str(acc))
        print('Precision: ' + str(prec))
        print('Recall: ' + str(rec))
        print('F1: ' + str(f1))
        print('Kappa: ' + str(kappa))
        
        scores = np.vstack((scores,np.array([[acc,prec,rec,f1,kappa]])))
        
        # Clean up memory
        if (transfer or ensemble):
            del sourceDataset, targetDataset, earlyStoppingDataset, validationDataset, classifications, labels
        else:
            del trainDataset, earlyStoppingDataset, validationDataset, classifications, labels
        
        # If specified, quit after single fold
        if ('singleFold' in kwargs) and (kwargs['singleFold'] is True):
            break
            
    # Print average performance
    print('Average Accuracy: ' + str(np.mean(scores[:,0],axis=0)) + u" \u00B1 " + str(np.std(scores[:,0],axis=0)))
    print('Average Precision: ' + str(np.mean(scores[:,1],axis=0)) + u" \u00B1 " + str(np.std(scores[:,1],axis=0)))
    print('Average Recall: ' + str(np.mean(scores[:,2],axis=0)) + u" \u00B1 " + str(np.std(scores[:,2],axis=0)))
    print('Average F1: ' + str(np.mean(scores[:,3],axis=0)) + u" \u00B1 " + str(np.std(scores[:,3],axis=0)))    
    print('Average Kappa: ' + str(np.mean(scores[:,4],axis=0)) + u" \u00B1 " + str(np.std(scores[:,4],axis=0)))
    
if (__name__ is "__main__") or (__name__ == "__main__"):
    runName = 'ganSHHS'
    dataSet = 'shhs'
    modelType = CNN_2D
    homeDir = './'

    numChannels = 1
    tune = False
    loadMdl = None #homeDir + 'clusterWsc/model'
    retrain = None #homeDir + 'getShhsModel/model'
    extractFeatures = utils.getSTFT #utils.stftAndCovariance
    loadDataFrom = None #homeDir + 'eegbud.npy'
    saveDataTo = None #homeDir + 'eegbud'
    augment = False
    sequential = True if modelType is CNN_LSTM else False
    adversarialInf = True if modelType is CNN_AdvInf else False
    multiInput = True if extractFeatures is utils.stftAndCovariance else False
    print('Training untuned GAN on entire SHHS dataset before using it to generate 50,000 artificial samples to pre-train a completely untrained CNN before re-training on 1 of 5 folds of sleep profiler data')
    print('Number of channels: ' + str(numChannels))
    
    transfer = False
    ensemble = False
    ensembleList = [homeDir + 'getCiccModel/model',homeDir + 'getWscModel/model',homeDir + 'getShhsModel/model',homeDir + 'getIsrucModel/model',homeDir + 'getMassModel/model',homeDir + 'getMrosModel/model']
    targetDataChannel = [["EEG3"]] #[["5","CH5","Ch5"]]
    targetDataDir = '/labs/cliffordlab/data/EEG/Sleep_Profiler' #'/labs/cliffordlab/data/EEG/hearables/Sleep_Data'
    transferMdl = None #homeDir + 'shhsFull/model'
    targetFeatureExtraction = lambda *args,**kwargs: utils.extractFromSleepProfiler(*args,**kwargs,extractFeatures=extractFeatures,dataFileList = dataFileList,debug = (True if dataSet is 'debug' else False))
    
    unsupervisedPretrain = None #AutoencoderPretrainer
    unlabeledFeatureExtraction = lambda *args,**kwargs: utils.extractEegBudUnlabeled(*args,**kwargs,extractFeatures=extractFeatures,debug = (True if dataSet is 'debug' else False))
    unlabeledDataChannel = [["5","CH5","Ch5"]]
    unlabeledDataDir = '/labs/cliffordlab/data/EEG/hearables/Sleep_Data'
    
    logDir = homeDir + 'logs/' + runName + '/'
    saveGeneratorTo = logDir + 'shhsGan'
    stages = {'n3':0,'n2':1,'n1':2,'rem':3,'awake':4,'undefined':5}
    max_trials = 40 # Note: this value only used when tuning
    epochs = 1000

    #validationSet = [] # Skip validation and use all data for training
    
    # Set settings according to the dataset being used
    if dataSet == 'debug':
        runName = 'debug' # Rename run name and log directory to avoid overwriting any currently running instances
        logDir = homeDir + 'logs/' + runName + '/'
        dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromWsc(*args,**kwargs,extractFeatures=extractFeatures,debug=True)
        possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        holdOut = []
        singleFold = True
        numFolds = 2

        # Override normal training parameters in order to speed up run
        epochs = 3
        max_trials = 3
    elif dataSet == 'cicc':
        dataDir = '/labs/cliffordlab/data/Challenge2018/training'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromCicc(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = ['C3-M2','C4-M1'][:numChannels]
        holdOut = []
        singleFold = False
        numFolds = 5
    elif dataSet == 'wsc':
        dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromWsc(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = [['C3_M2'],['O1_M2']][:numChannels]
        holdOut = []
        singleFold = False
        numFolds = 5
    elif dataSet == 'shhs':
        dataDir = '/labs/cliffordlab/data/shhs'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromShhs(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = [["EEG(sec)","EEG2","EEG 2","EEG sec"],["EEG","EEG1","EEG 1"]][:numChannels] # Multiple possible names for same possibleChannels
        holdOut = []
        singleFold = False
        numFolds = 5
    elif dataSet == 'SleepProfiler':
        dataDir = '/labs/cliffordlab/data/EEG/Sleep_Profiler'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSleepProfiler(*args,**kwargs,extractFeatures=extractFeatures,dataFileList = dataFileList)
        possibleChannels = [['EEG3'],['EEG2']][:numChannels]
        holdOut = [] #range(0,14,3)
        numFolds = 5 #14 - len(holdOut)
        singleFold = False
    elif dataSet == 'mass':
        dataDir = '/labs/cliffordlab/data/MASS'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMass(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = ['C3','C4'][:numChannels]
        holdOut = []
        numFolds = 5
        singleFold = False
    elif dataSet == 'isruc':
        dataDir = '/labs/cliffordlab/data/ISRUC'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromIsruc(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = [["C3-A2","C3-M2","C3"],["C4-A1","C4-M1","C4"]][:numChannels]
        holdOut = []
        numFolds = 5
        singleFold = False
    elif dataSet == 'ssc':
        dataDir = '/labs/cliffordlab/data/SSC'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromSSC(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = ["C3-A2"][:numChannels]
        holdOut = []
        numFolds = 5
        singleFold = False
    elif dataSet == 'mros':
        dataDir = '/labs/cliffordlab/data/MrOS'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromMrOs(*args,**kwargs,extractFeatures=extractFeatures)
        # NOTE: PossibleChannels labels indicate that this is a C3-A2 signal, but the MrOS document appears
        # to use 'A1' and 'A2' to mean the mastoids: https://sleepdata.org/datasets/mros/files/m/browser/documentation/MrOS_Visit1_PSG_Manual_of_Procedures.pdf
        possibleChannels = [["C3"],["C4"]][:numChannels]
        holdOut = []
        numFolds = 5
        singleFold = False
    elif dataSet == 'eegbud':
        dataDir = '/labs/cliffordlab/data/EEG/hearables/inhouse_sleep'
        featureExtractionFunc = lambda *args,**kwargs: utils.extractFromEegBudsAnn(*args,**kwargs,extractFeatures=extractFeatures)
        possibleChannels = [["5","CH5","Ch5","ERW-ELC","L-REarCanal"]][:numChannels] #1,2 or 5, ["C3-A2","C3-M2","EEG C3-A2","EEG C3-M2"]
        holdOut = []
        numFolds = 5
        singleFold = False

    else:
        print('Unknown dataset')

    if tune:
        # Override several parameters to speed up tuning
        singleFold = True
        numFolds = 4
        
    # If running via sbatch, don't show output
    if (__name__ == "__main__"):
        verbose = 0
    else:
        verbose = 2

    # Run main
    dataLoadFunc = lambda x: featureExtractionFunc(x,downSampleTo = 100, possibleChannels = possibleChannels, stages = stages)
    targetDataLoadFunc = lambda x: targetFeatureExtraction(x,downSampleTo = 100, possibleChannels = targetDataChannel, stages = stages)
    unlabeledDataLoadFunc = lambda x: unlabeledFeatureExtraction(x,downSampleTo = 100, possibleChannels = unlabeledDataChannel, stages = stages)
    main(epochs = epochs,verbose = verbose,saveMdl = logDir + 'model',logDir = logDir,dataLoadFunc = dataLoadFunc,saveDataTo=saveDataTo,loadDataFrom=loadDataFrom,sequential=sequential,
         max_trials = max_trials, dataDir = dataDir, singleFold = True, holdOut = holdOut, numFolds = numFolds, tune = tune, loadMdl = loadMdl,retrain = retrain,transfer = transfer,
         targetDataLoadFunc = targetDataLoadFunc, targetDataDir = targetDataDir, transferMdl = transferMdl,modelType = modelType,adversarialInf = adversarialInf,augment = augment,
         unsupervisedPretrain = unsupervisedPretrain,unlabeledDataLoadFunc = unlabeledDataLoadFunc,unlabeledDataDir = unlabeledDataDir,multiInput = multiInput,ensemble = ensemble,
         ensembleList = ensembleList,saveGeneratorTo = saveGeneratorTo)
    
