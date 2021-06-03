import numpy as np
import csv

#reading data from csv files.
with open('IrisGeometicFeatures_TrainingSet.txt') as csv_file:
    geomatic_training = csv.reader(csv_file, delimiter=',')


with open('IrisGeometicFeatures_TestingSet.txt') as csv_file:
    geomatic_test = csv.reader(csv_file, delimiter=',')


with open('IrisTextureFeatures_TrainingSet.txt') as csv_file:
    texture_training = csv.reader(csv_file, delimiter=',')


with open('IrisTextureFeatures_TestingSet.txt') as csv_file:
    texture_test = csv.reader(csv_file, delimiter=',')