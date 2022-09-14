import utils
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from transfertools.models import LocIT, CORAL
from tensorflow.keras.models import Model
from tensorflow.keras.backend import function
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential,losses
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, Conv2D, MaxPool2D, Flatten, Reshape, LSTM, GlobalAveragePooling2D, TimeDistributed, Masking, Lambda, Concatenate, Conv1D, MaxPooling1D, ReLU, Concatenate, Bidirectional, RNN, Add, StackedRNNCells, LSTMCell
from modCoral import RegCORAL, SCORAL

def tensor_MMD(x1,x2,gamma):
    # Caclculate maximum mean discrepancy between two observation tensors 'x1' and 'x2' 
    # using gaussian RBF kernel with gamma parameter 'gamma'
        
    # Randomly sample from the larger matrix so that both are the same size
    if x2.shape[0] > x1.shape[0]:
        idx = tf.random.shuffle(tf.range(x2.shape[0]),seed = 1)[:x1.shape[0]]
        x2 = tf.gather(x2,idx)
    elif x2.shape[0] < x1.shape[0]:
        idx = tf.random.shuffle(tf.range(x1.shape[0]),seed = 1)[:x2.shape[0]]
        x1 = tf.gather(x1,idx)
    
    # Define gaussian kernel between vectors
    K = lambda x,y: tf.math.exp(gamma*tf.math.reduce_sum(tf.math.square(x[:min([len(x),len(y)])] - y[:min([len(x),len(y)])]),axis=1))
    
    # Compute MMD
    mmd = tf.math.sqrt(2*(tf.math.reduce_mean(K(x1[0::2],x1[1::2])) + tf.math.reduce_mean(K(x2[0::2],x2[1::2])) - tf.math.reduce_mean(K(x1[0::2],x2[1::2])) - tf.math.reduce_mean(K(x2[0::2],x1[1::2]))))
    
    # Handle NaN
    if tf.math.is_nan(mmd): mmd = tf.zeros((1))
    
    return mmd


class MmdLoss(tf.keras.losses.Loss):
    def __init__(self,gamma = 1,weight = 1,**kwargs):
        self.gamma = gamma # gamma parameter for rbf kernel
        self.weight = weight # Weight to multiply the loss value by
        super().__init__(**kwargs)

        
    def call(self,labels,layer_out):
        # Loss function for minimizing MMD of layer outputs between source and target dataset
        
        if tf.math.reduce_all(labels == 1) or tf.math.reduce_all(labels == 0):
            # If all samples only from one dataset, return loss of 0
            return tf.zeros((1))
        
        # Get samples from each dataset
        src_out = tf.boolean_mask(layer_out,tf.squeeze(labels == 1))
        targ_out = tf.boolean_mask(layer_out,tf.squeeze(labels == 0))
        
        # Flatten each sample
        src_out = tf.reshape(src_out,[src_out.shape[0],tf.math.cumprod(src_out.shape[1:])[-1]])
        targ_out = tf.reshape(targ_out,[targ_out.shape[0],tf.math.cumprod(targ_out.shape[1:])[-1]])
        return self.weight*tensor_MMD(src_out,targ_out,self.gamma)

    

def denseNet(input_shape,classes=5):
    # Implementation of DenseNet
    print('Initializing DenseNet')
    
    inputs = Input(shape=input_shape)
    pretrainedMdl = tf.keras.applications.DenseNet121(include_top=False,weights=None,input_shape=input_shape)

    x = pretrainedMdl(inputs)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes,activation="softmax")(x)
    model = tf.keras.Model(inputs,outputs)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def deepSleepNet(inputShape = None,classes=5):
    # Load a pre-trained instance of Supratak et al's deepsleepnet from
    # a directory containing a .cpkt file

    # Construct model architecture
    inputs = Input(batch_shape=(1,)+inputShape)
    x1 = TimeDistributed(Conv1D(64,50,strides = 6,use_bias=False,padding="same"),name='l1_conv/conv1d')(inputs)
    x1 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l1_conv/bn')(x1)
    x1 = TimeDistributed(ReLU())(x1)
    x1 = TimeDistributed(MaxPooling1D(pool_size = 8,strides = 8,padding="same"))(x1)
    x1 = TimeDistributed(Dropout(.5))(x1)
    x1 = TimeDistributed(Conv1D(128,8,strides = 1,use_bias=False,padding="same"),name='l4_conv/conv1d')(x1)
    x1 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l4_conv/bn')(x1)
    x1 = TimeDistributed(ReLU())(x1)
    x1 = TimeDistributed(Conv1D(128,8,strides = 1,use_bias=False,padding="same"),name='l5_conv/conv1d')(x1)
    x1 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l5_conv/bn')(x1)
    x1 = TimeDistributed(ReLU())(x1)
    x1 = TimeDistributed(Conv1D(128,8,strides = 1,use_bias=False,padding="same"),name='l6_conv/conv1d')(x1)
    x1 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l6_conv/bn')(x1)
    x1 = TimeDistributed(ReLU())(x1)
    x1 = TimeDistributed(MaxPooling1D(pool_size = 4,strides = 4,padding="same"))(x1)
    x1 = TimeDistributed(Flatten())(x1)
    
    x2 = TimeDistributed(Conv1D(64,400,strides = 50,use_bias=False,padding="same"),name='l9_conv/conv1d')(inputs)
    x2 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l9_conv/bn')(x2)
    x2 = TimeDistributed(ReLU())(x2)
    x2 = TimeDistributed(MaxPooling1D(pool_size = 4,strides = 4,padding="same"))(x2)
    x2 = TimeDistributed(Dropout(.5))(x2)
    x2 = TimeDistributed(Conv1D(128,6,strides = 1,use_bias=False,padding="same"),name='l12_conv/conv1d')(x2)
    x2 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l12_conv/bn')(x2)
    x2 = TimeDistributed(ReLU())(x2)
    x2 = TimeDistributed(Conv1D(128,6,strides = 1,use_bias=False,padding="same"),name='l13_conv/conv1d')(x2)
    x2 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l13_conv/bn')(x2)
    x2 = TimeDistributed(ReLU())(x2)
    x2 = TimeDistributed(Conv1D(128,6,strides = 1,use_bias=False,padding="same"),name='l14_conv/conv1d')(x2)
    x2 = TimeDistributed(BatchNormalization(epsilon=1e-5),name='l14_conv/bn')(x2)
    x2 = TimeDistributed(ReLU())(x2)
    x2 = TimeDistributed(MaxPooling1D(pool_size = 2,strides = 2,padding="same"))(x2)
    x2 = TimeDistributed(Flatten())(x2)
    
    x = Concatenate(axis=-1)([x1,x2])
    x = TimeDistributed(Dropout(.5))(x)
    
    fc = TimeDistributed(Dense(1024),name='l19_fc/fc')(x)
    fc = TimeDistributed(BatchNormalization(),name='l19_fc/bn')(fc)
    fc = TimeDistributed(ReLU())(fc)

    lstmForwardCell = RNN(StackedRNNCells([tf.compat.v1.nn.rnn_cell.LSTMCell(512,use_peepholes=True,reuse=tf.compat.v1.get_variable_scope().reuse,name = 'fw/multi_rnn_cell/cell_' + str(i) + '/lstm_cell') for i in range(2)]),
                          return_sequences = True,stateful=True)
    lstmBackwardCell = RNN(StackedRNNCells([tf.compat.v1.nn.rnn_cell.LSTMCell(512,use_peepholes=True,reuse=tf.compat.v1.get_variable_scope().reuse,name = 'bw/multi_rnn_cell/cell_' + str(i) + '/lstm_cell') for i in range(2)]),
                           return_sequences = True,stateful=True,go_backwards=True)
    lstm = Bidirectional(lstmForwardCell,backward_layer=lstmBackwardCell,name = 'l21_bi_lstm/bidirectional_rnn')(x) #(x[:,:,None,:])
    lstm = TimeDistributed(Flatten())(lstm)
    
    combined = Add()([fc,lstm])
    combined = TimeDistributed(Flatten())(combined)
    combined = TimeDistributed(Dropout(.5))(combined)
    output = TimeDistributed(Dense(classes,activation='softmax'),name='l24_softmax_linear/fc')(combined)
    model = Model(inputs = inputs,outputs = output)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model
        
def CNN_2D(hp = None,classes=5,inputShape = None):
    # Create new CNN
    print('Initializing 2D CNN tuned for single-channel')
    
    # Define hyperparameters
    if hp is None: hp = kt.HyperParameters() # If not tuning, just use default for all hyperparameters
    conv_dropout = 0 #hp.Float('conv_dropout',0,.5,step=.02,default=.0)
    dense_dropout = hp.Float('dense_dropout',0,.9,step=.1,default=.5) #.6000000000000001
    l2_reg = tf.keras.regularizers.L2(l2=10**hp.Float('l2_regularization',-15,4,step=.1,default=-6.900000000000029)) #-14.9 
    glorot = tf.keras.initializers.GlorotUniform(seed=1)
    
    # Define model architecture
    model = Sequential([
        Conv2D(hp.Int('conv1_filters',80,200,default=135),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot,name='conv1'), #113
        MaxPool2D(),
        BatchNormalization(),
        Dropout(conv_dropout, seed=1),
        Conv2D(hp.Int('conv2_filters',80,200,default=79),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot,name='conv2'), #128
        MaxPool2D(),
        BatchNormalization(),
        Dropout(conv_dropout, seed=1),
        Conv2D(hp.Int('conv3_filters',80,200,default=86),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot,name='conv3'), #151
        MaxPool2D(),
        BatchNormalization(),
        Dropout(conv_dropout, seed=1),
        Conv2D(hp.Int('conv4_filters',80,200,default=146),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot,name='conv4'), #200
        MaxPool2D(),
        BatchNormalization(),
        Dropout(dense_dropout, seed=1),
        Flatten(),
        Dense(classes, activation='softmax',kernel_initializer=glorot)
        ])
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def CNN_MultiInput(hp = None,classes=5,stftShape = None,covShape = None):
    # Create new CNN
    print('Initializing 2D CNN')
    
    # Define hyperparameters
    if hp is None: hp = kt.HyperParameters() # If not tuning, just use default for all hyperparameters
    conv_dropout = 0 #hp.Float('conv_dropout',0,.5,step=.02,default=.0)
    dense_dropout = hp.Float('dense_dropout',0,.9,step=.1,default=.2)
    l2_reg = tf.keras.regularizers.L2(l2=10**hp.Float('l2_regularization',-15,4,step=.1,default=-5.600000000000033))
    glorot = tf.keras.initializers.GlorotUniform(seed=1)

    stftInputs = Input(shape = stftShape)
    covInputs = Input(shape = covShape)
    flatCovInputs = Flatten()(covInputs)
    x = Conv2D(hp.Int('conv1_filters',80,200,default=113),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(stftInputs)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv2_filters',80,200,default=123),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv3_filters',80,200,default=193),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv4_filters',80,200,default=149),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Concatenate(axis=-1)([x,flatCovInputs])
    x = BatchNormalization()(x)
    x = Dropout(dense_dropout, seed=1)(flatCovInputs)
    output = Dense(classes, activation='softmax',kernel_initializer=glorot)(x)

    model = Model(inputs = {'stft':stftInputs,'correlations':covInputs}, outputs = [output])

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model


@tf.custom_gradient
def grad_reverse(x):
    # Custom tensorflow operation which reverses the gradient sign for use in gradient reversal layer
    y = tf.identity(x) # On forward pass, return the inputs unchanged
    def custom_grad(dy):
        # On backwards pass, return the sign-reversed gradients
        return -dy 
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    # Gradient reversal layer for use in adversarial networks
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

def CNN_AdvInf(hp = None,classes=5,inputShape = None):
    # 2D CNN-based adversarial inference network
    print('Initializing CNN-AdvInf')

    # Define hyperparameters
    if hp is None: hp = kt.HyperParameters() # If not tuning, just use default for all hyperparameters
    conv_dropout = 0 #hp.Float('conv_dropout',0,.5,step=.02,default=.0)
    dense_dropout = 0 #hp.Float('dense_dropout',0,.9,step=.1,default=.5)
    l2_reg = tf.keras.regularizers.L2(l2=10**hp.Float('l2_regularization',-15,4,step=.1,default=-6.900000000000029))
    glorot = tf.keras.initializers.GlorotUniform(seed=1)

    # Define model architecture
    inputs = Input(shape = inputShape)
    x = Conv2D(hp.Int('conv1_filters',80,200,default=135),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(inputs)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv2_filters',80,200,default=79),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv3_filters',80,200,default=86),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(conv_dropout, seed=1)(x)
    x = Conv2D(hp.Int('conv4_filters',80,200,default=146),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot)(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dense_dropout, seed=1)(x)
    x = Flatten()(x)

    # Split model into two branches: one classifier and a descriminator which will try
    # to guess which dataset a sample belongs to.
    classifierOut = Dense(classes, activation='softmax',kernel_initializer=glorot,name='classifier')(x)
    x = GradReverse()(x)
    discriminatorOut = Dense(2, activation='softmax',kernel_initializer=glorot,name='descriminator')(x)

    model = Model(inputs=[inputs],outputs={"classifier":classifierOut,"descriminator":discriminatorOut})

    losses = {
	"classifier": "categorical_crossentropy",
	"descriminator": "categorical_crossentropy",
    }
    lossWeights = {"classifier": 1.0, "descriminator": 10**hp.Float('adversarialLossWeight',-10,-2,step=.1,default=-3)}

    model.compile(optimizer = 'adam',
                  loss = losses,
                  loss_weights=lossWeights,
                  metrics = ['accuracy'])

    return model


def FFN(hp = None,classes=5):
    # Create new FFN
    print('Initializing FFN')
    
    # Define hyperparameters
    if hp is None: hp = kt.HyperParameters() # If not tuning, just use default for all hyperparameters
    layer1_dropout = hp.Float('layer1_dropout',0,.8,step=.1,default=.4)
    layer2_dropout = hp.Float('layer2_dropout',0,.8,step=.1,default=.4)
    layer1_width = hp.Int('layer1_width',classes,40,default=28)
    layer2_width = hp.Int('layer2_width',classes,40,default=28)
    l2_reg = tf.keras.regularizers.L2(l2=10**hp.Float('l2_regularization',-10,8,step=.1,default=-10))
    
    # Define model architecture
    model = Sequential([Flatten(),
                        Dense(np.min([layer1_width,layer2_width]), activation='relu'),
                        Dropout(layer1_dropout),
                        Dense(np.max([layer1_width,layer2_width]), activation='relu'),
                        Dropout(layer2_dropout),
                        Dense(classes, activation='softmax')
                    ])
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model


def CNN_LSTM(hp = None,classes=5,inputShape = None):
    print('Initializing CNN_LSTM')
    
    # Define hyperparameters
    if hp is None: hp = kt.HyperParameters() # If not tuning, just use default for all hyperparameters
    conv_dropout = hp.Float('conv_dropout',0,.8,step=.1,default=0.8)
    lstm_dropout = hp.Float('lstm_dropout',0,.8,step=.1,default=0)
    dense_dropout = hp.Float('dense_dropout',0,.8,step=.1,default=.30000000000000004)
    l2_reg = tf.keras.regularizers.L2(l2=10**hp.Float('l2_regularization',-10,8,step=.1,default=-10))
    glorot = tf.keras.initializers.GlorotUniform(seed=1)
    
    # Define model architecture
    inputs = Input(shape = inputShape)
    x = TimeDistributed(Conv2D(hp.Int('conv1_filters',3,50,default=29),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot))(inputs)
    x = TimeDistributed(MaxPool2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(conv_dropout, seed=1))(x)
    x = TimeDistributed(Conv2D(hp.Int('conv2_filters',3,50,default=28),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot))(x)
    x = TimeDistributed(MaxPool2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(conv_dropout, seed=1))(x)
    x = TimeDistributed(Conv2D(hp.Int('conv3_filters',3,50,default=22),3,activation='relu',padding='same',kernel_regularizer=l2_reg, bias_regularizer=l2_reg,kernel_initializer=glorot))(x)
    x = TimeDistributed(MaxPool2D())(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dropout(lstm_dropout, seed=1))(x)
    x = LSTM(hp.Int('lstm_units',5,50,default=5),activation='relu',return_sequences=True,kernel_initializer=glorot)(x)
    x = TimeDistributed(Dropout(dense_dropout, seed=1))(x)
    output = TimeDistributed(Dense(classes, activation='softmax',kernel_initializer=glorot))(x)

    model = LongSequenceRNN(inputs,output)
        
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  sample_weight_mode='temporal',
                  weighted_metrics = ['accuracy'])
    return model

class LongSequenceRNN(tf.keras.Model):
    # An RNN which trains applies the gradient step every few samples in a
    # single sequence to reduce vanishing gradient problems on extremely long
    # sequences.
    
    ptBatchSize = 32

    def train_step(self, ptData):
        # Peforms gradient step on data from a single subject one small batch at a time
        x,y,weights = ptData

        numBatches = np.ceil(x.shape[1]/self.ptBatchSize).astype(int) # Calculate total batches per patient
        trainable_vars = self.trainable_variables

        for iBatch in range(numBatches):
            # Perform gradient step on each batch

            # Get data from batch
            batchSlice = slice(iBatch*self.ptBatchSize,np.min([(iBatch+1)*self.ptBatchSize,x.shape[1]]))
            xBatch = x[0:1,batchSlice]
            yBatch = y[0:1,batchSlice]
            weightsBatch = weights[0,batchSlice]
            
            with tf.GradientTape() as tape:
                y_predBatch = self(xBatch, training=True) # Forward pass

                # Compute loss/gradient step only on samples with non-zero weight
                loss = self.compiled_loss(yBatch[weightsBatch != 0],y_predBatch[weightsBatch != 0],regularization_losses=self.losses)

            # Compute gradients
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(yBatch[weightsBatch != 0],y_predBatch[weightsBatch != 0])

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class WeightedTrainModel:
    def __init__(self,mdl):
        self.mdl = mdl
    
    def fit(self,sourceDataset, targetDataset, validation_data = None, weights = {"source":.9,"target":.1}, **kwargs):
        addSrcWeights = lambda x,y: (x,y,weights["source"]) #(x,y,tf.numpy_function(weights["source"],tf.float32))
        addTargWeights = lambda x,y: (x,y,weights["target"]) #(x,y,tf.numpy_function(weights["target"],tf.float32))
        combinedDatasets = sourceDataset.unbatch() \
                                        .map(addSrcWeights) \
                                        .concatenate(targetDataset.unbatch().map(addTargWeights)) \
                                        .batch(32)
        self.mdl.fit(combinedDatasets,validation_data=validation_data,**kwargs)

    def predict(self,x):
        return self.mdl.predict(x)
        
class CoralTransfer:
    # Takes a tensorflow neural network 'mdl' and a CORAL domain transfer algorithm
    # 'transform' and outputs a model which performs transfer learning on the
    # network hidden layer activations at layer 'layerSignifier' (which can
    # be either a numerical index or string identifier).
    
    def __init__(self,mdl,transform = SCORAL(),layerSignifier = 0):

        print('Initializing CORAL transform model')
        self.origMdl = mdl
        self.transform = transform
        self.layerSignifier = layerSignifier
        self.scaler = StandardScaler()

    def fit(self,sourceDataset, targetDataset, validation_data = None, **kwargs):
        # Create transform between layer activations of source domain
        # After fitting transform retrain the neural network model on the
        # transformed target domain, in which case can pass all other keras
        # training parameters via kwargs.
        
        # Store dataset sizes for later use
        numSourceSamps = sourceDataset[0].shape[0] 
        numTargetSamps = targetDataset[0].shape[0] 
        
        if type(self.layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            index = None
            for idx, layer in enumerate(self.origMdl.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = self.layerSignifier
            
        layer = self.origMdl.layers[index] # Get hidden layer to extract activations from
        mdlUpperLayerList = [layer.layer if layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed else layer for layer in self.origMdl.layers[(index+1):]]
        self.mdlUpperLayers = Sequential(mdlUpperLayerList) # Make model from every subsequent layer
        self.timeDistributed = layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed
        
        # Construct function for obtainin hidden layer activations
        self.get_activations = function([self.origMdl.layers[0].input],[layer.output])
        
        # Obtain intermediate outputs for source and target domains
        featureDims = self.get_activations(sourceDataset[0][0])[0].shape[2:] 
        extractActivations = self.get_activations 
        a_t = np.vstack([extractActivations(batch)[0] for batch in targetDataset[0]]) 
        a_s = np.vstack([extractActivations(batch)[0] for batch in sourceDataset[0]]) 
        
        # Remove temporal dimension
        a_t = a_t.reshape((-1,) + featureDims) 
        a_s = a_s.reshape((-1,) + featureDims) 
        
        # Reshape arrays such that each sample is 1D
        reshaper = lambda x: x.reshape((1,-1)) 
        a_t_flat = np.vstack([reshaper(batch) for batch in a_t])
        a_s_flat = np.vstack([reshaper(batch) for batch in a_s])
        y_t = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in targetDataset[1]])
        y_s = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in sourceDataset[1]])
        
        # Ensure source and target dataset same size (required for domain transfer code)
        randomChoices = np.random.choice(a_s_flat.shape[0],a_t_flat.shape[0],replace=False)
        a_s_flat = a_s_flat[randomChoices]
        y_s = y_s[randomChoices]
        
        # Fit transform
        print('Training Domain Transformation')
        self.scaler.fit(a_t_flat.astype(np.double))
        self.transform.fit(Xs=a_s_flat,Xt=self.scaler.transform(a_t_flat),yt=np.argmax(y_t,axis=-1),ys=np.argmax(y_s,axis=-1))
        
        # Apply transform
        reverseReshape = lambda x: x.reshape(featureDims) 
        transformer = lambda x,y: self.transform.transfer(x,np.argmax(y,axis=-1)) 
        scaler = lambda x: self.scaler.transform(x) 
        a_t_transformed = np.vstack([reverseReshape(scaler(x[None,...])) for x in a_t_flat]) 
        a_s_transformed = np.vstack([reverseReshape(transformer(x[None,...],y)) for x,y in zip(a_s_flat,y_s)]) 
        
        # Combine source and target dataset
        trainData = np.concatenate((a_t_transformed,a_s_transformed))
        trainY = np.concatenate((y_t,y_s))

        # Shuffle source and target
        shuffledIndices = np.random.permutation(len(trainData))
        trainData = trainData[shuffledIndices]
        trainY = trainY[shuffledIndices]

        # Clean up memory
        del a_t_transformed, a_s_transformed, y_s, y_t
        
        if validation_data is not None:
            # Also transform early stopping set, if provided
            a_stopping = np.vstack([extractActivations(batch)[0] for batch in validation_data[0]]) #validation_data.map(extractActivations,tf.data.experimental.AUTOTUNE).unbatch()
            a_stopping = a_stopping.reshape((-1,) + featureDims) #a_stopping.unbatch() # Remove temporal dimension, if any
            y_stopping = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in validation_data[1]])
            a_stopping_transformed = np.vstack([reverseReshape(scaler(reshaper(batch))[None,...]) for batch in a_stopping])
        
        # Retrain model on transformed features
        print('Retraining model')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
        self.mdlUpperLayers.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy']) 
        
        self.mdlUpperLayers.fit(trainData,trainY,validation_data=[a_stopping_transformed,y_stopping],batch_size=32,**kwargs)
        
    def predict(self,x):
        # Predict labels for x using trained transform and model
        a_t = self.get_activations(x)[0]
        a_t_flat = a_t.reshape((a_t.shape[0]*a_t.shape[1],-1)) 
        a_t_scaled = self.scaler.transform(a_t_flat).reshape(a_t.shape)
        return self.mdlUpperLayers.predict(a_t_scaled[0])

    def splitModel(self,layerSignifier = None):
        # Extract the model activations
        # Note 1: layerSignifier argument not actually used and is only present to maintain
        # the same function signature as other models which do use it.
        
        # Create function for extracting features
        flattenTensor = lambda x: x.reshape((x.shape[0]*x.shape[1],-1)) #tf.reshape(x,[x.shape[0]*x.shape[1],-1]) if self.timeDistributed else lambda x: tf.reshape(x,[x.shape[0],-1]) # Helper function to flatten each activation tensor into a vector and remove temporal dimension, if necessary
        srcFeatureExtractor = lambda x,y: self.transform.transfer(flattenTensor(self.get_activations(x)[0]),y)
        targFeatureExtractor = lambda x,y: self.scaler.transform(flattenTensor(self.get_activations(x)[0]))
        
        classifier = lambda x: self.mdlUpperLayers.predict(np.reshape(x,(x.shape[0],) + self.mdlUpperLayers.input_shape[1:]))
        return (srcFeatureExtractor,targFeatureExtractor,classifier)

class HeadRetrain:
    # Simply retrains every layer above a specified layer in some pre-trained model with
    # no domain transfer or special transfer methods. Class created to improve modularity
    # by wrapping a standard tensorflow model in an object where the function signatures
    # are the same as in other transfer learning models.
    def __init__(self,mdl,layerSignifier = 0):
        print('Initializing head retrain model')
        self.origMdl = mdl
        self.layerSignifier = layerSignifier
        
    def fit(self,sourceDataset, targetDataset, validation_data = None, **kwargs):
        # Freeze every layer in a model except those above 'layerSignifier'
        # Note: 'sourceDataset' is unused.
        
        if type(self.layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            index = None
            for idx, layer in enumerate(self.origMdl.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = self.layerSignifier

        self.timeDistributed = self.origMdl.layers[index].__class__ is tf.python.keras.layers.wrappers.TimeDistributed

        layer = self.origMdl.layers[index] # Get hidden layer to extract activations from
        mdlUpperLayerList = [layer.layer if layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed else layer for layer in self.origMdl.layers[(index+1):]]
        self.mdlUpperLayers = Sequential(mdlUpperLayerList) # Make model from every subsequent layer
        self.timeDistributed = layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed

        # Construct function for obtainin hidden layer activations
        self.get_activations = function([self.origMdl.layers[0].input],[layer.output])

        # Obtain intermediate outputs for source and target domains
        featureDims = self.get_activations(sourceDataset[0][0])[0].shape[2:] 
        extractActivations = self.get_activations 
        a_t = np.vstack([extractActivations(batch)[0] for batch in targetDataset[0]])
        a_t = a_t.reshape((-1,) + featureDims) # Remove temporal dimension
        y_t = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in targetDataset[1]])

        if validation_data is not None:
            a_stopping = np.vstack([extractActivations(batch)[0] for batch in validation_data[0]])
            a_stopping = a_stopping.reshape((-1,) + featureDims)
            y_stopping = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in validation_data[1]])

        print('Retraining model')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
        self.mdlUpperLayers.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy']) 
        self.mdlUpperLayers.fit(a_t,y_t,validation_data=[a_stopping,y_stopping],batch_size=32,**kwargs)
        
        
    def splitModel(self,layerSignifier = None):
        # If regular tensorflow sequential model, create tensorflow function which extracts the learned features using the model
        if layerSignifier is None: layerSignifier = self.layerSignifier
        if type(layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            for idx, layer in enumerate(model.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = layerSignifier
            
        layer = self.origMdl.layers[index] # Get hidden layer to extract activations from
        activationExtractionFunction = function([self.origMdl.layers[0].input],[layer.output]) # Tensorflow function object to get activations from specified layer
        
        # Get feature extractor function
        flattenTensor = lambda x: tf.reshape(x,[x.shape[0]*x.shape[1],-1]) if layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed else lambda x: tf.reshape(x,[x.shape[0],-1]) # Helper function to flatten each activation tensor into a vector and remove temporal dimension, if necessary
        featureExtractor = lambda x,y: flattenTensor(activationExtractionFunction(x)[0]) # Return function for extracting vectors of activations
        
        # Get classifier
        classifier = lambda x: self.mdlUpperLayers.predict 
        
        return (featureExtractor,featureExtractor,classifier)
            
    def predict(self,x):
        # Predict labels for x using trained transform and model
        a_t = self.get_activations(x)[0].squeeze()
        return self.mdlUpperLayers.predict(a_t)
                
class TcaTransfer:
    # Takes a tensorflow neural network 'mdl' and a TCA domain transfer algorithm
    # 'transform' and outputs a model which performs transfer learning on the
    # network hidden layer activations at layer 'layerSignifier' (which can
    # be either a numerical index or string identifier).
    #
    # If n_components = 'no_change', dimensionality of TCA is chosen to avoid changing
    # the input size of the next layer
    # tcaKwargs arguments are all passed to the SSTCA initialization, and so follow
    # the same format and defaults listed here:
    # https://github.com/Vincent-Vercruyssen/transfertools/blob/master/transfertools/models/sstca.py
    
    def __init__(self,mdl,layerSignifier = -2,n_components = 'no_change',**tcaKwargs):
        print('Initializing TCA transform model')
        self.origMdl = mdl
        self.layerSignifier = layerSignifier
        self.n_components = n_components
        self.tcaKwargs = tcaKwargs

    def fit(self,sourceDataset, targetDataset, validation_data = None, **kwargs):
        # Create transform between layer activations of source domain
        # After fitting transform retrain the neural network model on the
        # transformed target domain, in which case can pass all other keras
        # training parameters via kwargs.

        # Store dataset sizes for later use
        numSourceSamps = utils.getNumBatches(sourceDataset)
        numTargetSamps = utils.getNumBatches(targetDataset)
        
        if type(self.layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            index = None
            for idx, layer in enumerate(self.origMdl.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = self.layerSignifier
            
        layer = self.origMdl.layers[index] # Get hidden layer to extract activations from
        
        
        if self.n_components is 'no_change': self.n_components = np.cumprod(layer.output_shape)[-1]
        self.transform = SSTCA(n_components=self.n_components,is_regress=False,**self.tcaKwargs)
        
        self.mdlUpperLayers = Sequential([Input(shape = [self.n_components]),
                                          Dropout(.5, seed=1),
                                          Dense(5, activation='softmax',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))]) #Sequential(self.origMdl.layers[(index+1):]) # Make model from every subsequent layer
        
        # Construct function for obtainin hidden layer activations
        self.get_activations = function([self.origMdl.layers[0].input],[layer.output]) # TODO: try function([self.origMdl.layers[0].input],[Reshape((-1,) + layer.output_shape[-1])(layer.output)])
        
        # Obtain intermediate outputs for source and target domains
        featureDims = self.get_activations(next(iter(sourceDataset))[0])[0].shape[1:]
        extractActivations = lambda x,y: (tf.numpy_function(self.get_activations,[x],tf.float32),y)
        a_t = targetDataset.map(extractActivations,tf.data.experimental.AUTOTUNE).unbatch()
        a_s = sourceDataset.map(extractActivations,tf.data.experimental.AUTOTUNE).unbatch()
        
        # Reshape arrays such that each sample is 1D
        reshaper = lambda x,y: (tf.reshape(x,(1,np.cumprod(featureDims)[-1])),y)
        a_t_flat = a_t.map(reshaper,tf.data.experimental.AUTOTUNE)
        a_s_flat = a_s.map(reshaper,tf.data.experimental.AUTOTUNE)
        a_t_np,y_t_np = utils.dataset2Numpy(a_t_flat)
        a_s_np,y_s_np = utils.dataset2Numpy(a_s_flat.shard(np.floor(numSourceSamps/numTargetSamps),0))
        
        # Ensure source and target dataset same size (required for domain transfer code)
        a_s_np = a_s_np[0:a_t_np.shape[0]]
        y_s_np = y_s_np[0:a_t_np.shape[0]]
        
        # Fit transform
        print('Training domain transformation')
        self.transform.fit(Xs=a_s_np,Xt=a_t_np,yt=np.argmax(y_t_np,axis=-1),ys=np.argmax(y_s_np,axis=-1))
        del a_s_np, a_t_np, y_s_np, y_t_np
        print('Domain transformation training complete')
        
        # Apply transform
        reverseReshape = lambda x,y: (tf.reshape(x,[-1]),y)
        targetTransformer = lambda x,y: (tf.numpy_function(self.transform.transfer_target,[x],tf.float32),y)
        sourceTransformer = lambda x,y: (tf.py_function(self.transform.transfer,[x],tf.float32),y)
        a_t_transformed = a_t_flat.map(targetTransformer,tf.data.experimental.AUTOTUNE).map(reverseReshape,tf.data.experimental.AUTOTUNE)
        a_s_transformed = a_s_flat.map(sourceTransformer,tf.data.experimental.AUTOTUNE).map(reverseReshape,tf.data.experimental.AUTOTUNE)
        del a_s_flat, a_t_flat
        
        # Combine source and target datasets
        trainDataset = a_s_transformed.concatenate(a_t_transformed).batch(32)
        del a_s_transformed, a_t_transformed
        
        if validation_data is not None:
            # Also transform early stopping set, if provided
            a_stopping = validation_data.map(extractActivations,tf.data.experimental.AUTOTUNE) \
                                         .unbatch() \
                                         .map(reshaper,tf.data.experimental.AUTOTUNE) \
                                         .map(targetTransformer,tf.data.experimental.AUTOTUNE) \
                                         .map(reverseReshape,tf.data.experimental.AUTOTUNE).batch(32)
        
        # Retrain model on transformed features
        print('Retraining model')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
        self.mdlUpperLayers.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy']) 
        self.mdlUpperLayers.fit(trainDataset,validation_data=a_stopping,**kwargs)
        
    def predict(self,x):
        # Predict labels for x using trained transform and model
        a_t = self.get_activations(x)[0]
        a_t_flat = a_t.reshape((a_t.shape[0],np.cumprod(a_t.shape[1:])[-1])) 
        a_t_transformed = self.transform.transfer_target(a_t_flat)
        return self.mdlUpperLayers.predict(a_t_transformed)
    
    def splitModel(self,layerSignifier = None):
        # Extract the model activations
        # Note: layerSignifier argument not actually used and is only present to maintain
        # the same function signature as other models which do use it.
        
        # Create function for extracting features
        flattenTensor = lambda x: tf.reshape(x,[x.shape[0],-1]) # Helper function to flatten each activation tensor into a vector
        srcFeatureExtractor = lambda x,y: self.transform.transfer(flattenTensor(self.get_activations(x)[0]))
        targFeatureExtractor = lambda x,y: self.transform.transfer_target(flattenTensor(self.get_activations(x)[0]))
        
        classifier = lambda x: self.mdlUpperLayers.predict(np.reshape(x,(x.shape[0],) + self.mdlUpperLayers.input_shape[1:]))
        return (srcFeatureExtractor,targFeatureExtractor,classifier)
    
class AutoencoderPretrainer(tf.keras.Model):
    # Builds an autoencoder from the first few layers of an existing model for the
    # purpose of unsupervised pre-training
    
    def __init__(self,baseModel,outputShape = None,tapLayer = -2,freezeBelow = -2):

        super(AutoencoderPretrainer, self).__init__()
        
        # Get index of layer to 'tap' and feed into decoder
        if type(tapLayer) == str:
            # Find index of layer if it is identified by it's name
            tapIndex = None
            for idx, layer in enumerate(baseModel.layers):
                if layer.name == layerName:
                    tapIndex = idx
                    break
        else:
            tapIndex = tapLayer
            
        # Get index of layer below which to make untrainable
        if type(freezeBelow) == str:
            # Find index of layer if it is identified by it's name
            freezeBelowIndex = None
            for idx, layer in enumerate(baseModel.layers):
                if layer.name == layerName:
                    freezeBelowIndex = idx
                    break
        else:
            freezeBelowIndex = freezeBelow
            
        # Freeze every layer below specified layer
        for layer in baseModel.layers[:freezeBelowIndex]: layer.trainable = False
        
        # Encoder is first few layers of base model
        encoder = baseModel.layers[:(tapIndex+1)]
        
        # Define decoder layers
        glorot = tf.keras.initializers.GlorotUniform(seed=1)
        l2_reg = tf.keras.regularizers.L2(l2=10**-6.900000000000029)
        decoder = [Flatten(),
                   Dense(np.cumprod(outputShape)[-1], activation='linear',kernel_initializer=glorot,kernel_regularizer=l2_reg, bias_regularizer=l2_reg),
                   Reshape(outputShape)]
        
        # Combine encoder and decoder
        self.autoencoder = Sequential(encoder + decoder)
        
    def call(self,x):
        return self.autoencoder(x)

class WeightedSoftVote:
    # Model used for collectively transferring a number of pre-trained models onto a sinlge task
    # and combining them via weighted soft vote.

    def __init__(self,inputShape,pretrainedList,architecture,transferMethod = None):
        print('Initializing ensemble model')

        self.transferMethod = transferMethod
        self.models = []
        for pretrainedModel in pretrainedList:
            # Load set of pre-trained models
            self.models.append(architecture())
            self.models[-1](np.zeros(inputShape)) # Forces model to be built
            self.models[-1].load_weights(pretrainedModel) # Load weights
            

    def predict(self,x):
        predictions = self.weights[0]*self.models[0].predict(x) # Initial run to get overall shape of predictions
        for i in range(1,len(self.models)): predictions = predictions + self.weights[i]*self.models[i].predict(x)
        return predictions

    def regressionObjective(self,weights,x,y):
        # Output residuals between predicted probability of the correct class
        # and the one-hot label of the correct class for use in optimization
        # of weights in weighted soft vote
        self.weights = weights/np.sum(weights)
        predictions = self.predict(x)
        return np.sum(predictions*y,axis=1)

    def fit(self,sourceDataset,targetDataset,**kwargs):
        
        for i in range(len(self.models)):
            # Re-train each individual model on the target set
            if self.transferMethod is None:
                # If no special transfer method is specified, perform simple retraining
                # of final dense layer
                for layer in model.layers[:-1]: layer.trainable = False
                self.models[i].fit(targetDataset,**kwargs)

            else:
                # Re-train models using specified transfer method
                self.models[i] = self.transferMethod(self.models[i]) # Encapsulate model in transfer learning object
                self.models[i].fit(sourceDataset,targetDataset,**kwargs)

        # Convert to numpy
        rawData = np.vstack(tuple([sample[0] for sample in targetDataset]))
        labels = np.vstack(tuple([sample[1] for sample in targetDataset]))

        # Fit weights for weighted soft vote
        #optimalWeights = least_squares(lambda weights: self.regressionObjective(weights,rawData,labels),
        #                               np.ones(len(self.models))/len(self.models),
        #                               method = 'trf')
        
        #self.weights = optimalWeights.x/np.sum(optimalWeights.x)
        self.weights = np.ones(len(self.models))/len(self.models)
        print('Voting weights: ' + str(self.weights))

    
class DeepDomainConfusion:
    # Model used for deep domain confusion

    def __init__(self,baseMdl,mmdLayerIdentifiers):
        print('Initializing deep domain adaptation network')
        self.baseMdl = baseMdl
        self.mmdLayerIdentifiers = mmdLayerIdentifiers

        # Assemble model
        input = Input(shape=self.baseMdl.layers[0].input_shape[1:])
        x = input
        outputs = {}
        losses = {}
        self.mmdLayers = []
        for i,iLayer in enumerate(self.baseMdl.layers):
            x = iLayer(x)
            if iLayer.name in self.mmdLayerIdentifiers or i in self.mmdLayerIdentifiers:
                # Obtain list of layers to use for outputs in calculating MMD loss
                self.mmdLayers.append(iLayer.name)
                outputs[iLayer.name] = x
                losses[iLayer.name] = MmdLoss(weight = .25)
        outputs['classifier'] = x
        losses['classifier'] = 'categorical_crossentropy'
        self.model = Model(inputs = input,outputs = outputs)
        
        
        self.model.compile(optimizer = 'adam',
                           loss = losses,
                           metrics = {'classifier':'accuracy'})
        self.model.run_eagerly = True

    def fit(self,sourceDataset, targetDataset, validation_data = None, **kwargs):
        # Train the domain adaptation model
        
        # Add labels to each dataset indicating whether the sample is from the target or source dataset (this is used
        # in the MMD calculation)
        relabelSrc = lambda x,y: (x,dict({layer:1 for layer in self.mmdLayers},**{'classifier':y}))
        relabelTarg = lambda x,y: (x,dict({layer:0 for layer in self.mmdLayers},**{'classifier':y}))
        relabeledSource = sourceDataset.unbatch().map(relabelSrc)
        relabeledTarg = targetDataset.unbatch().map(relabelTarg)
        
        # Combine datasets
        combinedDatasets = relabeledSource.concatenate(relabeledTarg)
        
        # Shuffle
        combinedDatasets = combinedDatasets.batch(256) \
                                           .shuffle(1000) \
                                           .unbatch() \
                                           .shuffle(1000) \
                                           .batch(128)
        
        if validation_data is not None:
            relabelValidation = lambda x,y: (x,{'classifier':y})
            validation_data = validation_data.map(relabelValidation)
        
        self.model.fit(combinedDatasets,validation_data = validation_data,**kwargs)

    def predict(self,x):
        return self.model.predict(x)

    def splitModel(self,layerSignifier = -2):
        # Extract activations at a particular layer
        if type(layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            for idx, layer in enumerate(self.model.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = layerSignifier
            
        layer = self.model.layers[index] # Get hidden layer to extract activations from
        activationExtractionFunction = function([self.model.layers[1].input],[layer.output]) # Tensorflow function object to get activations from specified layer
        
        # Get feature extractor function
        flattenTensor = lambda x: tf.reshape(x,[x.shape[0],-1]) # Helper function to flatten each activation tensor into a vector
        featureExtractor = lambda x,y: flattenTensor(activationExtractionFunction(x)[0]) # Return function for extracting vectors of activations
        
        # Get classifier
        mdlUpperLayers = Sequential(self.model.layers[(index+1):]) # Make model from every subsequent layer
        classifier = lambda x: mdlUpperLayers.predict(np.reshape(x,(x.shape[0],) + layer.get_output_shape_at(0)[1:]))
        return (featureExtractor,featureExtractor,classifier)
        
        
def trainUnsupervized(model,pretrainer,trainData,epochs,*args,earlyStoppingData = None,**kwargs):
    # Pretrain a model in an unsupervized task
    
    # Initialize the pre-training object
    pretrainerModel = pretrainer(model,*args,**kwargs)
    pretrainerModel.compile(optimizer='adam',loss=losses.MeanSquaredError())
    
    # Define callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
    
    # Train the model
    pretrainerModel.fit(trainData,validation_data=earlyStoppingData,epochs=epochs,shuffle=True,callbacks=callbacks)

class AdvInfEarlyStopping(tf.keras.callbacks.Callback):
    classifierLoss = {}
    descriminatorLoss = {}
    best_weights = None
    
    def __init__(self, patience=None, min_delta = 0.0):
        super(AdvInfEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
    
    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        self.descriminatorLoss[epoch] = logs['descriminator_loss']
        self.classifierLoss[epoch] = logs['classifier_loss']

        if self.patience and epoch > self.patience:
            # best weight if the current descriminatorLoss is less than epoch-patience descriminatorLoss. Simiarly for classifierLoss but when larger
            if self.descriminatorLoss[epoch] < (self.descriminatorLoss[epoch-self.patience] - self.min_delta) and self.classifierLoss[epoch] < (self.classifierLoss[epoch-self.patience] - self.min_delta):
                self.best_weights = self.model.get_weights()
            else:
                # to stop training
                print('Stopping Training Early')
                self.model.stop_training = True
                # Load the best weights
                self.model.set_weights(self.best_weights)
        else:
            # best weight are the current weights
            self.best_weights = self.model.get_weights()

class SubspaceAlignment:
    # Takes a tensorflow neural network 'mdl'  and outputs a model which
    # performs subspace alignment on between learned features from a source
    # and target dataset.
    # Based on paper "Unsupervised Visual Domain Adaptation Using Subspace
    # Alignment". Parameter names and default values are equal to those used
    # in original paper.
    # 
    # mdl ------------- Pre-trained CNN to use for transfer learning.
    # layerSignifier -- Index or name of layer above which all layers
    #                   will be re-trained.
    # d --------------- dimensionality of linear subspace. 

    def __init__(self,mdl,layerSignifier = -2,n_components = 100):
        print('Initializing subspace alignment model')
        self.origMdl = mdl
        self.layerSignifier = layerSignifier
        self.n_components = n_components

    def fit(self,sourceDataset, targetDataset, validation_data = None, **kwargs):
        # Perform the domain adapatation and re-train the model on the transformed
        # Source and target

        # Store dataset sizes for later use
        numSourceSamps = utils.getNumBatches(sourceDataset)
        numTargetSamps = utils.getNumBatches(targetDataset)
        
        if type(self.layerSignifier) == str:
            # Find index of layer if it is identified by it's name
            index = None
            for idx, layer in enumerate(self.origMdl.layers):
                if layer.name == layerName:
                    index = idx
                    break
        else:
            index = self.layerSignifier
            
        layer = self.origMdl.layers[index] # Get hidden layer to extract activations from
        self.timeDistributed = layer.__class__ is tf.python.keras.layers.wrappers.TimeDistributed
        
        # Make model to replace dense layer of pre-trained network
        self.mdlUpperLayers = Sequential([Input(shape = [self.n_components]),
                                          Dropout(.5, seed=1),
                                          Dense(5, activation='softmax',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))]) 

        # Construct function for obtainin hidden layer activations
        self.get_activations = function([self.origMdl.layers[0].input],[layer.output])
        
        # Obtain intermediate outputs for source and target domains
        featureDims = self.get_activations(sourceDataset[0][0])[0].shape[2:] 
        extractActivations = self.get_activations 
        a_t = np.vstack([extractActivations(batch)[0] for batch in targetDataset[0]]) 
        a_s = np.vstack([extractActivations(batch)[0] for batch in sourceDataset[0]]) 
        
        # Remove temporal dimension
        a_t = a_t.reshape((-1,) + featureDims) 
        a_s = a_s.reshape((-1,) + featureDims) 
            
        # Reshape arrays such that each sample is 1D
        reshaper = lambda x: x.reshape((1,-1)) 
        a_t_flat = np.vstack([reshaper(batch) for batch in a_t]) 
        del a_t
        a_s_flat = np.vstack([reshaper(batch) for batch in a_s]) 
        del a_s
        y_t = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in targetDataset[1]])
        y_s = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in sourceDataset[1]])
        
        # Train PCA
        self.scaler_s = StandardScaler()
        self.scaler_t = StandardScaler()
        pca_s = PCA(n_components = self.n_components)
        self.pca_t = PCA(n_components = self.n_components)
        a_s_scaled = self.scaler_s.fit_transform(a_s_flat)
        a_t_scaled = self.scaler_t.fit_transform(a_t_flat)
        pca_s.fit(a_s_scaled)
        a_t_transformed = self.pca_t.fit_transform(a_t_scaled)

        # Apply domain transform
        Xs = pca_s.components_.T
        Xt = self.pca_t.components_.T
        self.Xa = Xs@Xs.T@Xt
        a_s_transformed = a_s_scaled@self.Xa
        del a_s_scaled, a_t_scaled

        # Combine source and target dataset
        trainData = np.concatenate((a_t_transformed,a_s_transformed))
        trainY = np.concatenate((y_t,y_s))

        # Shuffle source and target
        shuffledIndices = np.random.permutation(len(trainData))
        trainData = trainData[shuffledIndices]
        trainY = trainY[shuffledIndices]

        # Clean up memory
        del a_t_transformed, a_s_transformed, y_s, y_t
        
        if validation_data is not None:
            # Also transform early stopping set, if provided
            a_stopping = np.vstack([extractActivations(batch) for batch in validation_data[0]]) 
            a_stopping = a_stopping.reshape((-1,) + featureDims) # Remove temporal dimension, if any
            a_stopping_flat = np.vstack([reshaper(batch) for batch in a_stopping])
            y_stopping = np.vstack([batch.reshape((-1,batch.shape[-1])) for batch in validation_data[1]])
            a_stopping_scaled = self.scaler_t.transform(a_stopping_flat)
            a_stopping_transformed = self.pca_t.transform(a_stopping_scaled)
            del a_stopping, a_stopping_flat, a_stopping_scaled
            
        # Retrain model on transformed features
        print('Retraining model')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, min_delta = .001, restore_best_weights = True)] # Perform early stopping
        self.mdlUpperLayers.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy']) 
        self.mdlUpperLayers.fit(trainData,trainY,validation_data=[a_stopping_transformed,y_stopping],batch_size=32,**kwargs) 
        
    def predict(self,x):
        # Predict labels for x using trained transform and model
        a_t = self.get_activations(x)[0]
        a_t_flat = a_t.reshape((a_t.shape[0]*a_t.shape[1],-1)) 
        a_t_transformed = self.pca_t.transform(self.scaler_t.transform(a_t_flat))
        return self.mdlUpperLayers.predict(a_t_transformed)
    
    def splitModel(self,layerSignifier = None):
        # Extract the model activations
        # Note: layerSignifier argument not actually used and is only present to maintain
        # the same function signature as other models which do use it.
        
        # Create function for extracting features
        flattenTensor = lambda x: x.reshape((x.shape[0]*x.shape[1],-1)) # Helper function to flatten each activation tensor into a vector and remove temporal dimension, if necessary
        srcFeatureExtractor = lambda x,y: self.scaler_s.transform(flattenTensor(self.get_activations(x)[0]))@self.Xa
        targFeatureExtractor = lambda x,y: self.pca_t.transform(self.scaler_t.transform(flattenTensor(self.get_activations(x)[0])))
        
        classifier = lambda x: self.mdlUpperLayers.predict(np.reshape(x,(x.shape[0],) + self.mdlUpperLayers.input_shape[1:]))
        return (srcFeatureExtractor,targFeatureExtractor,classifier)

