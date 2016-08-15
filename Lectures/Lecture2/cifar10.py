"""
CS231n Helper Function
DATE: 2016-08-04
AUTHOR: KYUNG MO KWEON
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

class CIFAR10(object):
  """Summary
  CIFAR 10 Initial Class
  
  Attributes:
      labels (list): CIFAR 10 Class labels
      path (str): PATH to CIFAR-10 Dataset
      test_dataset (list): test dataset file list
      train_datasets (list): train dataset file list
  """
  def __init__(self):
    """Summary
    """
    self.train_datasets = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    self.test_dataset = ["test_batch"]
    self.labels = []
    self.path = ""

  def load_single_batch(self, FILE):
    """Summary
    
    Args:
        FILE (str): File to Open
    
    Returns:
        data (np.array): file
    """
    with open(FILE, 'rb') as f:
      data = pickle.load(f, encoding='bytes')

    return data

  def load_multiple_batches(self, FILELIST):
    """Summary
    
    Args:
        FILELIST (list): Description
    
    Returns:
        TYPE: Description
    """
    X, Y = [], []
    for file in FILELIST:
      data = self.load_single_batch(file)
      X.append(data[b'data'])
      Y.append(data[b'labels'])

    return np.concatenate(X), np.concatenate(Y)


  def get_label(self, number):
    """Summary
    
    Args:
        number (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return self.labels[int(number)]

  def view_img(self, X):
    """Summary
    
    Args:
        X (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    plt.imshow(X.reshape(3, 32, 32).transpose(1, 2, 0))
    
  def load_CIFAR10(self, PATH):
    """Summary
    
    Args:
        PATH (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    self.path = PATH
    self.train_datasets = [os.path.join(PATH, file) for file in self.train_datasets]
    self.test_dataset = [os.path.join(PATH, file) for file in self.test_dataset]
    self.labels = self.load_single_batch(os.path.join(PATH, "batches.meta"))
    Xtr, Ytr = self.load_multiple_batches(self.train_datasets)
    Xte, Yte = self.load_multiple_batches(self.test_dataset)

    return Xtr, Ytr, Xte, Yte


if __name__ == "__main__":
  c10 = CIFAR10()
  dataset_path = "./cifar-10-batches-py/"

  Xtr, Ytr, Xte, Yte = c10.load_CIFAR10(dataset_path)


  print("Xtr Shape:{}".format(Xtr.shape))
  print("Ytr Shape:{}".format(Ytr.shape))
  print("Xte Shape:{}".format(Xte.shape))
  print("Yte Shape:{}".format(Yte.shape))