import gzip
import io
from urllib.request import urlopen
from pathlib import Path


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


dr = DigitRecognition()
dr.loadRawFiles()
