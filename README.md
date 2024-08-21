# A fruits classification model

---
This repo contains an implementation of fruits classification based on Convolutional Neural Network using Keras, visualizing by scikit-learn, matplot for plotting accuracy curves and seaborn for plotting confusion matrixes, interactiving by jupyter notebook.
The dataset comes from fruit-360[https://www.kaggle.com/datasets/moltean/fruits]. Special thanks for it.

## Usage

---
Note if you wan to retrain the model, dataset should be placed into a closet folder named `input` in root directory or you can change the file path encoded in source code.

```bash
git clone https://github.com/MsRCAtN/fruit-360-recognition-cnn.git && cd fruit-360-recognition-cnn
pip3 install -r requirements.txt
cd src && jupyter notebook init.ipynb 
```

## Architecture

---

This model contains 3 convolution layers and 3 pooling layers for feature extraction, the kernel size of convolution layer was set to 3x3 with ReLU as activation function.
Each pooling layer have a dropout layer appends after in order to prevent overfiting. Dropout rate was set to 35%.
Then flatten the output from convolution layer for fed into fully connected layer. A dense layer as fully connected layer with ReLU activation function to reduce overfitting. A dropout layer with rate of 50% is appended.
Finally a dense layer with softmax as activation function as output layer.

## Evaluation

---
In the preset conditions, the model has 97% validating accuracy and 95% testing accuracy. However, if you want to try it with different conditions, you can change the epochs and batch size in the code.

## Citation

---
The model is trained on the [fruit-360](https://www.kaggle.com/datasets/moltean/fruits) dataset.
