# Transfer Learning Comparison
Code used in paper [Comparison of Deep Transfer Learning Algorithms and Transferability Measures for Wearable Sleep Staging](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-022-01033-3) published in *Biomedical Engineering Online* on September 12, 2022

### Setup
If you have not already done so, download following datasets: Computing in Cardiology Challenge 2018 (CiCC); Sleep Heart Health Study (SHHS); Wisconsin Sleep Cohort (WSC); Montreal Archive of Sleep Studies (MASS); Institute of Systems and Robotics, University of Coimbra (ISRUC); and Osteoporotic Fractures in Men Study dataset (MrOS). After downloading all datasets, change the file paths to each dataset to on lines 62–97 in cnn_model/compareTransferability.py and 81–116 in deepsleepnet/compareTransferability.py.

### Running Testing on CNN Models
All pretrained models are included in the repo. The trained CNN models are located in cnn_model/ and named in the format <dataset name>Cnn/. To run the testing on the convolutional architecture, set the source dataset, pre-trained model and output file name in cnn_model/compareTransferability.py to the desired source dataset, pre-trained model, and name for the output file, respectively. This can be done by changing the value of ```srcDataSet```, ```savedMdlDir```, and ```saveOutputTo``` on lines 31–34. Set ```retrainLayers``` on line 36 to select the number of layers to retrain (setting this value to -4 will retrain the dense layer and preceeding batch norm layer, setting it to -8 will additionally retrain the adjacent convolutional layer and preceeding batch norm layer). The following shell command can then run the code:
```shell
python3 cnn_model/compareTransferability.py
```
This will cause the model to be re-trained using all transfer learning algorithms using the designated source dataset on the target dataset, then evaluated and the results written to an .xls file. Note that depending on the speed of your machine, this could take several days to run in full.

### Running Testing on Deepsleepnet Models
As with the CNN models, all pretrained models are included in the repo. The trained DeepSleepNet models are located in deepsleepnet/ and named in the format <dataset name>Deepsleepnet/kerasMdl/. As with testing on the CNN architecture, testing on the Deepsleepnet architectre can be run by first setting the source dataset, pre-trained model and output file name in deepsleepnet/compareTransferability.py to the desired source dataset, pre-trained model, and name for the output file, respectively. This can be done by changing the value of ```srcDataSet```, ```savedMdlDir```, and ```saveOutputTo``` on lines 33–36. Then run the shell command:
```shell
python3 cnn_model/compareTransferability.py
```
All results will be saved to the .xls file. If you are running the code multiple times but don't want to resume from a checkpoint, make sure to delete the *_cpkt.txt file before running.

### Regenerating CNN Models
Training the CNN models on the source datasets from scratch can be done by setting ```dataSet``` on line 389 of cnn_model/sleep_staging_main.py to the desired dataset and setting ```runName``` on line 388 to the desired output directory and running 
```shell
python3 cnn_model/sleep_staging_main.py
```
The trained model will be saved to the location specified in ```runName```.

### Regenerating CNN Models
Training the Deepsleepnet models on the source datasets requires running a modified version of the [Deepsleepnet code](https://github.com/akaraspt/deepsleepnet), which is described by Supratak *et al* [here](https://ieeexplore.ieee.org/document/7961240). In deepsleepnet/deepsleepnet/LOO_crossval.py, set the value ```dataset``` on line 9 to the desired dataset and ```output_dir``` on line 10 to the desired output directory and run 
```shell
python3 deepsleepnet/deepsleepnet/LOO_crossval.py
```
The trained model will output to ```output_dir```/kerasMdl.
