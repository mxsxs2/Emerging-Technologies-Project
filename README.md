# Emerging-Technologies-Project
Set of notebooks of Python numpy.random, MNIST, Iris and Digit recognition script

# Content
* [Prerequisite](#Prerequisite)
* [How to run](#How-to-run-the-script)
* Jupyter notebooks
  * [Numpy random](#Numpy-random)
  * [Iris dataset](#Iris-dataset)
  * [MNIST database](#MNIST-database)
  * [Notebook explaining the digit recognition script](#Digit-recognition)
# Prerequisite
This project relies on python 3.6. The easiest way to acquire python and all the dependecnies for the notebooks and digit recognition script is to download anaconda from [here](https://www.anaconda.com/download/).

The project was developed on windows. It works on windows and has not been tested on any other platform.

Anaconda will install all the dependencies except `keras` and `tensorflow`.
* to install install `tensorflow` write in command line ```conda install -c conda-forge tensorflow```
    * [Tensorflow](https://www.tensorflow.org/) is the underlying machine learning framework for keras.
* to install install `keras` write in command line ```conda install -c conda-forge keras ```
    * [Keras] is the deep learning library used for the digit recognition script.  

If the abowe two will not install, they are bot awailable with [Anaconda Navigator](https://anaconda.org/anaconda/anaconda-navigator) 
# How to run the script
digitrec.py [-h] [--model {keras,knn}] [--verbose]
                           [--checkaccuracy] [--limit] [--image IMAGE]

Optional arguments:
*  ```-h```, ```--help```    Show this help message and exit
*  ```--model {keras,knn,mlpc,gaussian,svc}``` The model to use. One of: kreas, knn. Default is keras
*  ```--verbose```           If flag exist, extra informations is provided about MNIST files
*  ```--checkaccuracy```     If flag exist, the trained model will be checked for accuracy
*  ```--limit```             If flag exist, the model will use only 1000 records to train and test. This does not apply for keras!
*  ```--savemodel```         Save the trained model into a file to speed up the application run for next time.
*  ```--loadmodel```         Load trained model from file. This will disregard the `--model` attribute
*  ```--image```             Path for an image to recognise the number from. It can take a directory path with images in it. If a direcotry path is supplied the last / has to be omitted

The output of the application can be piped into a file the following way:
```python digitrec.py > output.txt```

## Jupyter Notebooks
Jupyter notebook is a simple way to present code snippets with explanations. It supports markdown, latex and various programming languages. The default language is python3.
### Open notebooks
The notebooks can be viewed or edited locally once Jyputer is installed. Jupyter comes with Anaconda.
Clone this repository and in the library run ```jupyter notebook``` this will open up a new browser window where the notebooks are visible. 

### Numpy random
[Numpy.random](https://docs.scipy.org/doc/numpy-1.15.1/reference/routines.random.html) is a collection of methods for random number generations, permutations and number distributions created by [numpy](http://www.numpy.org/). 

[Link for notebook](/notebooks/Nnumpy-random.ipynb)
### Iris dataset
The dataset is a collection of samples from three types of Iris flowers. This dataset is commonly used for statistical classification and starter machine learning projects (both supervised and unsupervised)

The dataset contains 150 samples evenly devided between the tree types of flower. The meauserments are taken from the flowers are:
* sepal length
* sepal width
* petal length
* petal width

[Link for notebook](/notebooks/iris-datatset.ipynb)

[Source 1](https://en.wikipedia.org/wiki/Iris_flower_data_set)
[Source 2](https://archive.ics.uci.edu/ml/datasets/iris)
[Source 3](https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/pandas-with-iris.ipynb)
[Source 4](https://www.ritchieng.com/machine-learning-iris-dataset/)

### MNIST database
The MNIST database is a collection of handwritten images which are used to train a neural network to recognize digits from a paper. MNIST also provides a database for testing the neural network.

The training imeages file contains 60000 images and the test images file contains 10000 images. These files are in a special format therefore they have to be read byte by byte.

[Link for notebook](/notebooks/mnist-dataset.ipynb)

[Link for MNIST database documentation](http://yann.lecun.com/exdb/mnist/)
#### Digit recognition
I developed a digit recognition script to  train a choice of neuronetwork to recognize hand written numbers from images. Once the netwrok is trained it tries to predict a digit from a provided source.

[Link for notebook](/notebooks/digit-recognition.ipynb)