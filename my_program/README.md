# Using KERAS API for training

## Main program use jupyter notebook file (**main.ipynb**)
* model.summary()  
Inspect the shape of layer  
* model.compile()  
Define optimizer and loss function  
* model.fit()  
Train the Unet model of DeepCGH  
* model.predict() and load_model()  
If already trained model, we can use 'load_model' to predict CGH

## Other file
* Create data generator (**data.py**) for training  
Add debug_generator for training  
Add debug_sample for prediction  

* Create custom loss (**loss.py**) for training  
Using subclass to create custom loss

* Create DeepCGH model (**model.py**)  
return KERAS functional model

## Setup
* Anaconda on Windows 10  
* Python 3.8.5  
* CUDA 11 + cudnn v8.0.5.39  

## Requirements  
* tensorflow 2.4  
* numpy 1.19.5  
* scikit-image  
