import cv2 as cv
import matplotlib.pyplot as plt
import torch
import os
import glob
import torchvision
from torchvision import transforms

def show_example_image():
  img = cv.imread('data/images/JPCLN001.jpg', cv.IMREAD_GRAYSCALE)
  mask = cv.imread('data/labels/JPCLN001.png', cv.IMREAD_GRAYSCALE)

  plt.figure(figsize=(10, 10))
  plt.subplot(1, 2, 1)
  plt.imshow(img, cmap='gray')
  plt.title('Image')
  plt.axis('off')
  plt.subplot(1, 2, 2)
  plt.imshow(mask, cmap='gray')
  plt.title('Mask')
  plt.axis('off')
  plt.show()

def train_dataset_grades(n):
  """
  x1 - [0..10] - number of hours spent studying
  x2 - [1..5] - average grade in high school
  x3 - [0..10] - number of absences per high school year
  y - 0 or 1 - attained bachelor's degree
  """
  x1 = 10 * torch.rand(n)         # [0..10]
  x2 = 4 * torch.rand(n) + 1      # [1..5]
  x3 = 10 * torch.rand(n)         # [0..10]
  y = 0.8*x1 + 0.2*x2 - 0.5*x3 + 0.1
  y = (y > 0).float()
  return torch.stack([x1, x2, x3], dim=1), y

def test_dataset_grades(n):
  return train_dataset_grades(n)

def get_image_files_in_folder(folder):
  jpgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
  if len(jpgs) == 0:
    pngs = sorted(glob.glob(os.path.join(folder, '*.png')))
    return pngs
  return jpgs

def get_mnist_dataset_train():
  train_dataset = torchvision.datasets.MNIST(
    root='.', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
  )
  return train_dataset

def get_mnist_dataset_test():
  test_dataset = torchvision.datasets.MNIST(
    root='.', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
  )
  return test_dataset

def dice_loss(inputs, targets):
  inputs = inputs.view(-1)
  targets = targets.view(-1)
  smooth = 1.
  
  intersection = (inputs * targets).sum()                            
  dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
  
  return 1 - dice