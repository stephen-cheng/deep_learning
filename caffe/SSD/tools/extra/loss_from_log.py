#!/usr/bin/env python
# Modified base on Martin Kersner's script: https://github.com/martinkersner/train-CRF-RNN/blob/master/loss_from_log.py

from __future__ import print_function
import sys
import re
import numpy as np
import matplotlib.pyplot as plt

from utils import strstr

def main():
  log_files = process_arguments(sys.argv)

  train_iteration = []
  train_loss      = []
  lr              = []
  test_iteration  = []
  test_loss       = []
  test_accuracy   = []

  top1_accuracy   = []
  top5_accuracy   = []

  base_test_iter  = 0
  base_train_iter = 0

  for log_file in log_files:
    with open(log_file, 'rb') as f:
      if len(train_iteration) != 0:
        base_train_iter = train_iteration[-1]
        base_test_iter = test_iteration[-1]

      for line in f:
        # TRAIN NET
        if strstr(line, 'Iteration') and strstr(line, 'lr'):
          matched = match_iteration(line)
          train_iteration.append(int(matched.group(1)))
          matched = match_lr(line)
          lr.append(float(matched.group(1)))

        elif strstr(line, 'Train net output'):
          matched = match_loss(line)
          train_loss.append(float(matched.group(1)))

        # TEST NET
        elif strstr(line, 'Iteration') and strstr(line, 'Testing net'):
          matched = match_iteration(line)
          test_iteration.append(int(matched.group(1)))

        elif strstr(line, 'Test net output #2'):
          matched = match_loss(line)
          test_loss.append(float(matched.group(1)))

        elif strstr(line, 'Test net output #0'):
          matched = match_top1(line)
          top1_accuracy.append(float(matched.group(1)))

        elif strstr(line, 'Test net output #1'):
          matched = match_top5(line)
          top5_accuracy.append(float(matched.group(1)))

  print("TRAIN", train_iteration, train_loss)
  print("TEST", test_iteration, test_loss)
  print("LEARNING RATE", train_iteration, lr)
  print("TOP1_ACCURACY", test_iteration, top1_accuracy)
  print("TOP5_ACCURACY", test_iteration, top5_accuracy)

  # loss
  plt.plot(train_iteration, train_loss, 'k', label='Train loss')
  plt.plot(test_iteration, test_loss, 'r', label='Test loss')
  plt.legend()
  plt.ylabel('Loss')
  plt.xlabel('Number of iterations')
  plt.savefig('loss.png')

  plt.show()

  # learning rate
  plt.clf()
  plt.plot(train_iteration, lr, 'g', label='Learning rate')
  plt.legend()
  plt.ylabel('Learning rate')
  plt.xlabel('Number of iterations')
  plt.savefig('lr.png')

  plt.show()
  
  # evaluation
  plt.clf()
  plt.plot(test_iteration, top1_accuracy, 'm', label='Top-1 accuracy')
  plt.plot(test_iteration, top5_accuracy, 'c', label='Top-5 accuracy')
  plt.legend(loc=0)
  plt.savefig('evaluation.png')

  plt.show()


def match_iteration(line):
  return re.search(r'Iteration (.*),', line)

def match_loss(line):
  return re.search(r'loss = (.*) \(', line)

def match_lr(line):
  return re.search(r'lr = (.*)', line)

def match_top1(line):
  return re.search(r'acc = (.*)', line)
  
def match_top5(line):
  return re.search(r'acc5 = (.*)', line)

def process_arguments(argv):
  if len(argv) < 2:
    help()

  log_files = argv[1:]
  return log_files

def help():
  print('Usage: python loss_from_log.py [LOG_FILE]+\n'
        'LOG_FILE is text file containing log produced by caffe.'
        'At least one LOG_FILE has to be specified.'
        'Files has to be given in correct order (the oldest logs as the first ones).'
        , file=sys.stderr)

  exit()

if __name__ == '__main__':
  main()
