import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import Sequential
from tensorflow.keras.backend import function
from scipy.spatial.distance import pdist, squareform

def splitModel(model,layerSignifier = -2):
    # Function to split model into feature extractor and classifier at the layer specified either by a name or index
    
    if (model.__class__ is tf.python.keras.engine.sequential.Sequential) or (model.__class__ is tf.python.keras.engine.functional.Functional):
        # If regular tensorflow sequential or functional model, create tensorflow function which extracts the learned features using the model
        if type(layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            for idx, layer in enumerate(model.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = layerSignifier
            
        layer = model.layers[index] # Get hidden layer to extract activations from
        activationExtractionFunction = function([model.layers[0].input],[layer.output]) # Tensorflow function object to get activations from specified layer
        
        # Get feature extractor function
        flattenTensor = (lambda x: tf.reshape(x,[x.shape[0]*x.shape[1],-1])) if layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed else (lambda x: tf.reshape(x,[x.shape[0],-1])) # Helper function to flatten each activation tensor into a vector
        featureExtractor = lambda x: flattenTensor(activationExtractionFunction(x)[0]) # Return function for extracting vectors of activations
        
        # Get classifier
        upperLayers = Sequential([layer.layer if layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed else layer for layer in model.layers[(index+1):]]) # Make model from every subsequent layer (removing TimeDistributed wrapper if necessary)
        classifier = lambda x: upperLayers.predict(x)
        
        return (featureExtractor,classifier)
    
    else:
        # If some type of transfer model, call that model's built-in model splitting function
        return model.splitModel(layerSignifier)

def MMD(x1,x2,gamma):
    # Caclculate maximum mean discrepancy between two observation matrices 'x1' and 'x2' 
    # using gaussian RBF kernel with gamma parameter 'gamma'
    
    # Shuffle
    np.random.shuffle(x1)
    np.random.shuffle(x2)
    
    # Randomly sample from the larger matrix so that both are the same size
    if x2.shape[0] > x1.shape[0]:
        x2 = x2[np.random.choice(x2.shape[0], x1.shape[0], replace=False), :]
    elif x2.shape[0] < x1.shape[0]:
        x1 = x1[np.random.choice(x1.shape[0], x2.shape[0], replace=False), :]
    
    # Define gaussian kernel between vectors
    K = lambda x,y: np.exp(gamma*np.sum(np.square(x[:min([len(x),len(y)])] - y[:min([len(x),len(y)])]),axis=1))
    
    # Compute MMD
    return np.sqrt(2*(np.mean(K(x1[0::2],x1[1::2])) + np.mean(K(x2[0::2],x2[1::2])) - np.mean(K(x1[0::2],x2[1::2])) - np.mean(K(x2[0::2],x1[1::2]))))

def hypothesis_margin(x,y,shrink = True):
    # compute the hypothesis margin of samples x with class labels y

    if shrink:
        # Downsample to reduce memory usage
        x = np.float32(x[::10])
        y = y[::10]
    
    totalMargin = 0 # Track the total margin
    distances = squareform(pdist(x)) # Compute euclidean distance between each sample
    mask = np.full(x.shape[0], True, dtype=bool) 
    for i in range(x.shape[0]):
        mask[i] = False # Mask to prevent the distance from a point to itself from being included in search
        nearhitDist = np.min(distances[i,np.logical_and(y == y[i],mask)]) # Distance to nearest point in same class
        nearmissDist = np.min(distances[i,y != y[i]]) # Distance to nearest point in same class
        
        # Tally total margin 
        # Note: the actual value of hypothesis margin should be .5*(||x[i] - nearmiss|| - ||x[i] - nearhit||) but
        # the .5 is not multiplied until later for the sake of efficiency
        totalMargin = totalMargin + np.abs(nearmissDist - nearhitDist) 
        mask[i] = True # Reset mask for use in next iteration
        
    return .5*totalMargin/x.shape[0]

def h_score(targFeatures,y):
    """
    Compute the H-score of a set of learned features with labels y extracted from the target 
    set using a model pre-trained on the source set.
    
    H-score is a measure of the transferability of a pre-trained model f onto some target task defined in 
    Yajie Bao's "An Information-Theoretic Approach to Transferability in Task Transfer Learning".
    Note that H-score is not the same as 'transferability' which is defined in the same paper. Transferability
    is H-score normalized by the theoretical highest possible H-score achievable for a given target and so can 
    be considered a measure of the ease-of-transfer-learning which does not vary by the target. H-score, however,
    varies by target and so it is not valid to compare the H-score on one target to the H-score on another target. 
    The advantage of using H-score instead of transferability is simply that H-score is faster to compute.
    """
    
    # Remove constant columns because these don't contribute to classification or but cause the covariance matrix to be singular
    cleanFeatures = targFeatures[:, ~np.all(targFeatures[1:] == targFeatures[:-1], axis=0)]
    
    covF = np.cov(cleanFeatures,rowvar=False) # Computes cov(f(x))
    
    possibleLabels = np.unique(y)
    conditionalMeans = np.zeros((possibleLabels.shape[0],cleanFeatures.shape[1]))
    conditionalMeans[:,:] = np.NaN # Filling with NaN allows later error checking
    
    # Compute E[f(x)|y] for each possible value of y
    for i,label in enumerate(possibleLabels):
        conditionalMeans[i] = np.mean(cleanFeatures[y == label],axis = 0)
    
    # Sanity check - presence of nan values suggests some data was not added matrix
    assert not np.isnan(conditionalMeans).any(), 'Unknown Error: matrix contains nan values'
    
    # Compute cov(E[f(x)|y])
    conditionalMeansCov = np.cov(conditionalMeans,rowvar=False)
    
    return np.trace(np.linalg.inv(covF) @ conditionalMeansCov)

def LEEP(x,y,model):
    """
    A measure of transferability of a pre-trained tensorflow model onto a target dataset with features x and lables y.
    First developed in 'LEEP: A New Measure to Evaluate Transferability of Learned Representations'
    """
    z = model(x) # Probabilities for dummy labels on the target dataset
    classes = np.unique(y)
    
    # Calculate joint distribution p(y,z)
    jointPDist = np.zeros((classes.shape[0],z.shape[1]))
    for i in range(classes.shape[0]):
        jointPDist[i] = np.mean(z[y == classes[i]],axis=0)
    
    assert not np.isnan(jointPDist).any(), 'Error: jointPDist contains NaN values. Check if model has NaN-valued weights or if there are NaN-valued features.'
    
    # compute marginal distribution p(z)
    marginalPDist = np.mean(z,axis=0)
    
    # compute conditional distribution p(y|z)
    conditionalPDist = np.zeros((classes.shape[0],z.shape[1]))
    for i in range(classes.shape[0]):
        jointPDist[i] = jointPDist[classes[i]]/marginalPDist
    
    # Compute LEEP
    perClassLeep = np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        perClassLeep[i] = np.log(np.sum(z[y == classes[i]] @ jointPDist[i,None].T))
        
    return np.mean(perClassLeep)

def TDAS(source,target,d,epsilon):

    # Train PCA
    scaler_source = StandardScaler()
    scaler_target = StandardScaler()
    pca_source = PCA(n_components = d)
    pca_target = PCA(n_components = d)
    
    source_scaled = scaler_source.fit_transform(source)
    target_scaled = scaler_target.fit_transform(target)
    pca_source.fit(source_scaled)
    pca_target.fit(target_scaled)

    # Obtain bases
    Xs = pca_source.components_.T
    Xt = pca_target.components_.T

    # Get sim() value for each pair of vectors in each dataset
    A = Xs@Xs.T@Xt@Xt.T
    sims = source_scaled@A@target_scaled.T

    # Count number of elements in target which are within radius epsilon of each sample in source
    withinRadius = np.sum(sims <= epsilon,axis = 1)
    return np.mean(withinRadius)
    
