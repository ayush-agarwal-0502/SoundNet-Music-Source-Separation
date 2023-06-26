# Instrument-Separator-UNet
A supervised, deep learning project using the U-Net model to accurately separate individual musical instruments from audio recordings.

* _Teammates_ : Ayush Agarwal , Akshita Gupta
* _Project domain_ : Machine Learning (ML) , Deep Learning (DL) , Natural Language Processing (NLP)
* _Tools Used_ : Tensorflow , Python , Jupyter Notebook
* _Domain Knowledge_ : Music Source Separation problem , U-Nets

........................................![image](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/3df2ba23-bd80-4d45-93c3-e2b4ca92a6c0)
![image](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/4add7e84-e0ba-43d5-b895-6b3334a0c0a3)
![image](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/f17657dc-139d-487f-875c-9b716957f7c2)

## About the project - Problem Statement and our INNOVATION

After some research , we found out that most of the present solutions that exist for the source separation problem were based on taking the Fourier Transform of the audio , studying the characteristic frequency spectrum of a particular source type , then masking that particular band out . I was aware from my experience in computer vision already that U-Nets have been used since a long time (ok few years) for Image Segmentation problems , but could not find much work upon U-Nets being used over audio datasets . Since in a way thinking about source separation problem , we are segmenting the different frequency bands of the audio sample , hence I felt that U-Nets could be potentially useful for source separation problem too . Hence we decided to give our own novel and naive innovation an attempt . 

## Demo results :

Here is the original song that was fed into the neural network:
https://github.com/AkGu2002/Instrument-Separator-UNet/blob/master/Al%20James%20-%20Schoolboy%20Facination.stem.mp4


 Here is the result we achieved:
 https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/5ecf4698-e618-4d65-82da-3142fe9611d0
 


 ..................![image](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/7721612c-3384-4cf7-b0b1-27f9b007db9a)


## Initial Challenges Faced
Initially we tried to train the ML model over the whole musdb dataset (dataset explanation given later) , however there were RAM memory limit excedeed errors in the kaggle notebooks . So we had to train and test the UNet over just 2 songs (which is still millions of samples by the way) . Also these kinds of errors limited our ability to make the ML model wider , thus the width of model is just 64 timesteps , thus limiting the ability of our ML model to use the information spread spacially across time (each second the song has about 10000 samples , and we are using only 64 , think about it , could even we as humans point out separate sources in this less time variation of a sound ) . 

Our dataset had stereo audio - what it means is that audio would be different for left and right ear . However to reduce the computational complexity and yet keep the training stable , we decided to use mono-audio instead of stereo audio . We used left part of audio files as training dataset and right part as test dataset . 

Our future work could involve making the U-Net wider , so that it can capture the time variation of the signal more accurately . 

## Our Solution :

### Input
##### *MUSDB 2018 Dataset*

..........................![image](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/048d70f7-a010-408c-a044-a94da168993e)

The musdb18 consists of 150 songs of different styles along with the images of their constitutive objects.

It contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs. Supervised approaches were used to train on the training set and test on both sets. 

All files from the musdb18 dataset are encoded in the Native Instruments stems format (.mp4). It is a multitrack format composed of 5 stereo streams, each one encoded in AAC @256kbps. These signals correspond to:

0 - The mixture,

1 - The drums,

2 - The bass,

3 - The rest of the accompaniment,

4 - The vocals.

For each file, the mixture correspond to the sum of all the signals. All signals are stereophonic and encoded at 44.1kHz.

### Dependency/ Library Used
numpy as np

pandas as pd 

tensorflow as tf 

keras

matplotlib

IPython

### Preprocessing the Input Dataset
Overall, the code processes the audio data from the mus_train dataset by dividing it into smaller chunks of 64 time stamps, pairs each chunk with the corresponding drums data, and stores them in a pandas DataFrame for further analysis or training in a machine learning model. 

As the audio tracks are stereophonic, therefore the left channel of the audio tracks in mus_train has been used as train dataset and the right channel of the audio tracks has been used as validation set. To make the process less time consuming, the model has been trained for 2 tracks only. 

## Constructing the Model
#### What is U-Net?
The U-Net model is a convolutional neural network (CNN) architecture commonly used for image segmentation tasks. The U-Net architecture derives its name from its U-shaped design, consisting of an encoding path (contracting path) and a decoding path (expansive path). The encoding path captures high-level features by progressively reducing the spatial resolution through convolutional and pooling layers. The decoding path, which is symmetric to the encoding path, upsamples the feature maps using deconvolutional layers to restore the spatial resolution. 

The U-Net architecture effectively combines the advantages of a CNN's feature extraction capability with the ability to preserve spatial information. It has been widely adopted in various image segmentation tasks, such as medical image analysis (e.g., tumor segmentation, cell segmentation), semantic segmentation, and more recently, even in audio-based tasks like instrument separation.

##### U-Net Model Structure
![Screenshot 2023-06-23 165902](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/f47a823f-6b69-4468-9881-0028913cabd9)

### Model Constructed in this Project

* The input layer takes in audio data with variable-length sequences (shape: (None, 64)).

* The model starts with a series of convolutional layers (Conv1D) with increasing filters (16, 32, 64, 128, 256, 512), each followed by batch normalization and leaky ReLU activation.

* Then, the model performs upsampling (transposed convolution) using Conv1DTranspose layers to restore the spatial resolution.

* At each upsampling stage, skip connections are implemented by concatenating the upsampled features with the corresponding features from the encoding path using Concatenate layers.

* ReLU activation is applied after each concatenation.
* Dropout layers with a dropout rate of 0.5 are used after the first and third upsampling stages.
* Finally, a Conv1DTranspose layer with 1 filter is used to generate a mask, and the output layer applies element-wise multiplication (Multiply) between the input audio and the mask.

### Training the Model

* The loss function is specified as Mean Absolute Error ('mae'), which measures the average absolute difference between the predicted and target values. 
* The optimizer chosen is Adam, with a learning rate of 1e-3.

* The training data (x_train and y_train) and validation data (x_val and y_val) are prepared by converting them into TensorFlow tensors. The df_train and df_test DataFrames are used to extract the respective input (x) and target (y) data. 

* The epochs parameter is set to 40, indicating the number of times the model will iterate over the entire training dataset.

* The verbose parameter is set to 2 to display a detailed progress bar, providing information about the training and validation steps.

### Plotting the Graph of Train_Error and Val_Error with each Epoch
![Screenshot 2023-06-23 173204](https://github.com/AkGu2002/Instrument-Separator-UNet/assets/74046369/db46b0e7-72e1-4fa7-8f55-12b7a0f9dfea)


The model is tested on the 11th track of mus_test. The original track and the predicted track have been uploaded :)



