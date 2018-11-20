import gzip
import io
from urllib.request import urlopen
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


class DigitRecognition:

    def __init__(self, verbose: bool = True, load_test_data: bool = True, check_accuracy: bool = True):
        """
        Constructor of class.
        Loads the training and test files
        verbose: extra details are printed about the files
        load_test_data: to load the validation data into memory or not
        check_accuracy: to check the accuracy of the machine learning method or not
        """
        self.verbose = verbose
        self.load_test_data = load_test_data
        self.check_accuracy = check_accuracy
        # Load the files
        self.__loadRawFiles()
        # Read the files into arrays
        self.__readTrainingAndTestData()

    def prepareMachineLearningModel(self, model, limited: bool):
        """
        Trains a machine learnong model and calculates accuracy score.
        Prediction is done with 100 test items if limited is on.
        When limited is off then prediction will be done with 10000 items. It can take long time to finish 
        """
        # Check if the model has precit method
        predict = getattr(model, "predict", None)
        if callable(predict):
            # Set the current model to previous model
            self.model = model
            # Train the model
            print("\nTraining model :", model)
            model.fit(self.train_images, self.train_labels)
            print("Model trained")
            if self.load_test_data and self.check_accuracy:
                print("Predicting")
                # Predict test data
                pred = model.predict(
                    self.test_images[0: 100] if limited == True else self.test_images)
                print("Calculating accuracy score")
                # Calculate accuracy
                acc_knn = accuracy_score(
                    self.test_labels[0: 100] if limited == True else self.test_labels, pred)
                print("Model prediction accuracy: %f" % acc_knn)
        else:
            print("Provided model does not have predict method!")

    def predictWithPreviousModel(self, toBePredicted):
        """
        Tryes to predict a number from the input.
        If there a model is not set yet. It raises and exception
        """
        # Check if a model was loaded
        if hasattr(self, 'model'):
            predicted = self.model.predict(toBePredicted)
            print("Predicted: ", predicted)
            return predicted
        else:
            raise Exception(
                "Call prepareMachineLearningModel before predictWithPreviousModel")

    def imageAsArray(self, image_name: str):
        """
        Reads in a picture from a file and return it as a 784 size pixel array consiting with numbers from 0 to 1
        """
        # Open the image and convert to gray scale and finally invert it so it maches with the training data set
        i = ImageOps.invert(Image.open(image_name).convert('L'))
        # Resize the image the same size as the training images
        i = i.resize((28, 28))
        # Convert to array, flaten it out and normalise it
        i_arr = np.asarray(i).reshape(784)/255
        return i_arr

    def __openOrDownloadAndOpenGzipFile(self, file_name: str, url: str)->str:
        """
        Checks if the given file exists, if it doesnt then downloads from the given url
        """
        if self.verbose:
            print('Checking %s' % file_name)

        file = Path(file_name)
        # Check if the file exist
        if(file.is_file() != True):
            print('Downloading %s' % url)
            # Download and read if not
            to_read = urlopen(url).read()
        else:
            # Read if it does
            to_read = file.read_bytes()
        # unzip the file
        with gzip.open(io.BytesIO(to_read), 'rb') as f:
            return f.read()

    def __loadRawFiles(self)->None:
        """
        Unzip the training and test data
        """
        self.train_images = self.__openOrDownloadAndOpenGzipFile(
            'data/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        self.train_labels = self.__openOrDownloadAndOpenGzipFile(
            'data/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        if self.load_test_data:
            self.test_images = self.__openOrDownloadAndOpenGzipFile(
                'data/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
            self.test_labels = self.__openOrDownloadAndOpenGzipFile(
                'data/t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        print("Data files loaded.")

    def __readTrainingAndTestData(self):
        """
        Reads in both training and test data
        """
        print("\nReading training images and labels into arrays")
        self.train_images, self.train_labels, train_item_number = self.__readImagesAndLabels(
            self.train_images, self.train_labels)
        if self.load_test_data:
            print("\nReading test images and labels into arrays")
            self.test_images, self.test_labels, test_item_number = self.__readImagesAndLabels(
                self.test_images, self.test_labels)

    def __readImagesAndLabels(self, images_raw, labels_raw):
        """
        Reads the images and labels into arrays
        """
        # Get det images and label meta data
        pictures_number, columns_number, rows_number, label_number = self.__getImageAndLabelMetaData(
            images_raw, labels_raw)
        # Check if the numbers of images and labels match
        if pictures_number != label_number:
            raise Exception("The number of pictures and labels does not mach!")
        # Read the images and labels into arrays
        images, labels = self.__loadPicturesAndLabelsToArrays(
            images_raw, pictures_number, columns_number, rows_number, 16, labels_raw, 8)

        return images, labels, pictures_number

    def __getImageAndLabelMetaData(self, images_raw, labels_raw):
        """
        Checks if the magic numbers are correct and reads in the first set of bites of the files.
        """
        if self.verbose:
            print("Meta data of pictures file:")

        # Confirm if the first four byteis 2051
        is_it_the_right_bytes = int.from_bytes(
            images_raw[0:4], byteorder='big') == 2051
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST pictures file")
        if self.verbose:
            print('Is the magic number correct: %r' % is_it_the_right_bytes)

        # Number of pictures should be from bytes 4 to 8 and should be read in big endian
        pictures_number = int.from_bytes(
            images_raw[4:8], byteorder='big')
        if self.verbose:
            print('Number of pictures: %d' % pictures_number)

        # Number of rows should be from 8 to 12
        rows_number = int.from_bytes(
            images_raw[8:12], byteorder='big')
        if self.verbose:
            print('Number of rows: %d' % rows_number)

        # Number of columns should be from 12 to 16
        columns_number = int.from_bytes(
            images_raw[12:16], byteorder='big')
        if self.verbose:
            print('Number of columns: %d' % columns_number)
        if self.verbose:
            print("Meta data of labels file:")

        # Confirm if the first four byteis 2049
        is_it_the_right_bytes = int.from_bytes(
            labels_raw[0:4], byteorder='big') == 2049
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST labels file")
        if self.verbose:
            print('Is the magic number correct: %r' % is_it_the_right_bytes)

        # Number of pictures should be from bytes 4 to 8 and should be read in big endian
        label_number = int.from_bytes(
            labels_raw[4:8], byteorder='big')
        if self.verbose:
            print('Number of Labels: %d' % label_number)
        return pictures_number, columns_number, rows_number, label_number

    def __loadPicturesAndLabelsToArrays(self, picture_file_content, pictures_number: int, columns_number: int, rows_number: int, pictures_offset: int, label_file_content, labels_offset: int):
        """
        Loads a set of pictures and labels into two arrays.
        The number of pictures and labels has to match
        The method does not reads in each picture flat as columns_number*rows_number.
        """
        # Set up an array for picture storage
        pictures = np.zeros(
            (pictures_number, columns_number*rows_number), dtype=float)
        labels = np.zeros(
            (pictures_number), dtype=int)
        # The current picture 1-59999
        current_image = 0
        # The iteration index
        i = pictures_offset
        print("Converting images and labels to array. Number of items: %d" %
              pictures_number)
        # Run a loop until the end of the byte array
        while i < len(picture_file_content):
            # Convert a row to float types and normalise it for better machine learning performance
            a = np.frombuffer(
                picture_file_content[i:i+columns_number*rows_number], dtype=np.uint8)
            # Set the current picture
            pictures[current_image] = a
            # Normalise the numbers to be between 0 and 1
            pictures[current_image] /= 255
            # Read in the label for this image
            labels[current_image] = int.from_bytes(
                label_file_content[current_image+labels_offset:current_image+labels_offset+1], byteorder='big')
            # Go to the next picture
            current_image += 1
            # Increase the counter with the size of the columns
            i += columns_number*rows_number
        return pictures, labels


#dr = DigitRecognition(False, False, False)
dr = DigitRecognition(True, True, True)
dr.prepareMachineLearningModel(KNeighborsClassifier(), True)
# dr.predictWithPreviousModel([dr.test_images[9999]])
# print(dr.test_labels[9999])
dr.predictWithPreviousModel([dr.imageAsArray('data/5.jpg')])
