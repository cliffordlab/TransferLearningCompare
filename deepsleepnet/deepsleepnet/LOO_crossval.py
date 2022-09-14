from train import train
from predict import predict
from glob import glob
import sys
sys.path.append('..')
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

dataset = 'mros'
output_dir = './output_mros'
n_folds = 2
fold_idx = 0
pretrain_epochs = 100
finetune_epochs = 200

if dataset == 'debug':
    dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
    output_dir = output_dir + '_debug/'
    dataFileList = glob(dataDir + '/*.edf')[:3]
    n_folds = 3
    pretrain_epochs = 20
    finetune_epochs = 20
    
elif dataset == 'cicc':
    dataDir = '/labs/cliffordlab/data/Challenge2018/training'
    dataFileList = glob(dataDir + '/*/')
elif dataset == 'wsc':
    dataDir = '/labs/cliffordlab/data/WSC/polysomnography/visit1'
    dataFileList = glob(dataDir + '/*.edf')
elif dataset == 'shhs':
    dataDir = '/labs/cliffordlab/data/shhs'
    dataFileList = glob(dataDir + '/edfs/shhs1/*.edf')
elif dataset == 'SleepProfiler':
    dataDir = '/labs/cliffordlab/data/EEG/Sleep_Profiler'
    dataFileList = glob(dataDir + '/*.edf')
elif dataset == 'mass':
    dataDir = '/labs/cliffordlab/data/MASS'
    dataFileList = glob(dataDir + '/*/*PSG.edf')
elif dataset == 'isruc':
    dataDir = '/labs/cliffordlab/data/ISRUC'
    dataFileList = (glob(dataDir + '/subgroupI/*/*.rec') + glob(dataDir + '/subgroupII/*/*1.rec') + glob(dataDir + '/subgroupIII/*/*1.rec'))
elif dataset == 'mros':
    dataDir = '/labs/cliffordlab/data/MrOS'
    dataFileList = glob(dataDir + '/polysomnography/edfs/visit1/*.edf')
else:
    print('Unknown dataset')

for fold in range(fold_idx,n_folds):
    # Note: this code currently only runs a single fold which trains on the entire dataset
    testFiles = [] #dataFileList[fold::n_folds]
    trainFiles = list(set(dataFileList) - set(testFiles))
    finetuned_model_path = train(trainFiles,dataset,output_dir,pretrain_epochs = pretrain_epochs,finetune_epochs = finetune_epochs)
    #print("Training complete. Testing model...")
    #predict(testFiles,dataset,finetuned_model_path,output_dir)
    break
