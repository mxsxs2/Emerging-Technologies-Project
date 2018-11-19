import gzip
import io
from urllib.request import urlopen
from pathlib import Path
import numpy as np


class DigitRecognition:

    def openOrDownloadAndOpenGzipFile(self, file_name, url):
        """
        Checks if the given file exists, if it doesnt then downloads from the given url
        """
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

    def loadRawFiles(self):
        """
        Unzip the training and test data
        """
        self.train_images_raw = self.openOrDownloadAndOpenGzipFile(
            'data/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        self.train_labels_raw = self.openOrDownloadAndOpenGzipFile(
            'data/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        self.test_images_raw = self.openOrDownloadAndOpenGzipFile(
            'data/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
        self.test_labels_raw = self.openOrDownloadAndOpenGzipFile(
            'data/t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        print("Train and test data loaded.")

    def readImagesAndLabels(self, images_raw, labels_raw):
        """
        Reads the images and labels into arrays
        """
        # Get det images and label meta data
        pictures_number, columns_number, rows_number, label_number = self.getImageAndLabelMetaData(
            images_raw, labels_raw)
        # Check if the numbers of images and labels match
        if pictures_number != label_number:
            raise Exception("The number of pictures and labels does not mach!")
        # Read the images and labels into arrays
        images, labels = self.loadPicturesAndLabelsToArrays(
            images_raw, pictures_number, columns_number, rows_number, 16, labels_raw, 8)

        return images, labels, pictures_number

    def readTrainingAndTestData(self):
        """
        Reads in both training and test data
        """
        print("\nReading training images and labels into arrays")
        self.train_images, self.train_labels, train_item_number = self.readImagesAndLabels(
            self.train_images_raw, self.train_labels_raw)
        print("\nReading test images and labels into arrays")
        self.test_images, self.test_labels, test_item_number = self.readImagesAndLabels(
            self.test_images_raw, self.test_labels_raw)

    def getImageAndLabelMetaData(self, images_raw, labels_raw):
        """
        Checks if the magic numbers are correct and reads in the first set of bites of the files.
        """
        print("\nMeta data of pictures file:")
        # Confirm if the first four byteis 2051
        is_it_the_right_bytes = int.from_bytes(
            images_raw[0:4], byteorder='big') == 2051
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST pictures file")
        print('Is the magic number correct: %r' % is_it_the_right_bytes)
        # Number of pictures should be from bytes 4 to 8 and should be read in big endian
        pictures_number = int.from_bytes(
            images_raw[4:8], byteorder='big')
        print('Number of pictures: %d' % pictures_number)
        # Number of rows should be from 8 to 12
        rows_number = int.from_bytes(
            images_raw[8:12], byteorder='big')
        print('Number of rows: %d' % rows_number)
        # Number of columns should be from 12 to 16
        columns_number = int.from_bytes(
            images_raw[12:16], byteorder='big')
        print('Number of columns: %d' % columns_number)

        print("Meta data of labels file:")
        # Confirm if the first four byteis 2049
        is_it_the_right_bytes = int.from_bytes(
            labels_raw[0:4], byteorder='big') == 2049
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST labels file")
        print('Is the magic number correct: %r' % is_it_the_right_bytes)
        # Number of pictures should be from bytes 4 to 8 and should be read in big endian
        label_number = int.from_bytes(
            labels_raw[4:8], byteorder='big')
        print('Number of Labels: %d' % label_number)
        return pictures_number, columns_number, rows_number, label_number

    def loadPicturesAndLabelsToArrays(self, picture_file_content, pictures_number, columns_number, rows_number, pictures_offset, label_file_content, labels_offset):
        """
        Loads a set of pictures and labels into two arrays.
        The number of pictures and labels has to match
        """
        # Set up an array for picture storage
        pictures = np.zeros(
            (pictures_number, rows_number, columns_number), dtype=int)
        labels = np.zeros(
            (pictures_number), dtype=int)
        # The current row a picture 1-28
        current_row = 1
        # The current picture 1-59999
        current_image = 0
        # The iteration index
        i = pictures_offset
        print("Converting images and labels to array. Number of items: %d" %
              pictures_number)
        # Run a loop until the end of the byte array
        while i < len(picture_file_content):
            # Convert a row to int types
            a = [c for c in picture_file_content[i:i+columns_number]]
            # Set the row the current picture
            pictures[current_image][current_row-1] = a
            # Go to next row
            current_row += 1
            # If the current row is the same as the size of the rows
            if(current_row > rows_number):
                # Set the row to number 1
                current_row = 1
                # Read in the label for this image
                labels[current_image] = int.from_bytes(
                    label_file_content[current_image+labels_offset:current_image+labels_offset+1], byteorder='big')
                # Go to the next picture
                current_image += 1

            # Increase the counter with the size of the columns
            i += columns_number
        return pictures, labels


dr = DigitRecognition()
dr.loadRawFiles()
dr.readTrainingAndTestData()
