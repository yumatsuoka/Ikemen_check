#IkemenCheck:Regression or Classification with CNN
##Abstract
regression or classification with Convolutional Neural Networks using TensorFlow.  

In 'src/cnn/.', "get_image_tensor.py" is to make a object of images dataset and classification.py and regression_cnn are CNNs codes.  
In 'src/implement_to_datasets', "*.sh" or "*.py" are to transform images.  

##Requirements
Implement CNNs  
-Python 2.x (checked Python 2.7.6)  
-TensorFlow(checked TensorFlow 0.6.0)  
-Pillow(checked Pillow 3.0.0)  
-numpy(checked numpy 1.11.0)  
-pandas(checked pandas 0.17.1)  

Implement transforming images  
-OpenCV(checked OpenCV 3.0)  
-pandas(checked pandas 0.17.1)  

##How to use
-prepare datasets(.jpg) and target_data(.csv) with 'src/implement_to_datasets/.'scripts  
-python classification.py or regression_cnn.py
