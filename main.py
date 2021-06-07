import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from numpy import testing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def learning_one_file(attributeLength, classLength, trainingFile, testingFile):
    train_data, result_data = read_one_file(trainingFile)

    test_data, test_result = read_one_file(testingFile)

    model = Sequential()
    model.add(Dense(12, input_dim=attributeLength, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(classLength, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, result_data, epochs=150, batch_size=64)

    _, accuracy = model.evaluate(test_data, test_result)
    print('Accuracy: %.2f' % (accuracy * 100))


def learning_two_files(attributeLength, classLength, training_file1,testing_file1,training_file2,testing_file2):    
    train_data, result_data = read_two_files(training_file1,training_file2)

    test_data, test_result = read_two_files(testing_file1,testing_file2)

    model = Sequential()
    model.add(Dense(12, input_dim=attributeLength, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(classLength, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_data, result_data, epochs=150, batch_size=64)

    _, accuracy = model.evaluate(test_data, test_result)
    print('Accuracy: %.2f' % (accuracy * 100))


def read_one_file(filename):
    dataset = list()
    resultset=list()
    skip = ['@RELATION', '@ATTRIBUTE', '@DATA']
    with open(filename) as file:
        for line in file:
            if not any(x in line for x in skip):
                result = [x.strip() for x in line.split(',')]
                if len(result) > 1:
                    dataset.append(result[:len(result) - 1])
                    resultset.append([result[len(result) - 1]])
    
    dataset = np.array(dataset,dtype=np.float)
    result = np.array(resultset)

    #sc = StandardScaler()
    #dataset = sc.fit_transform(dataset)

    ohe = OneHotEncoder()
    resultset = ohe.fit_transform(resultset).toarray()

    return [dataset,resultset]

def read_two_files(filename1,filename2):
    dataset1 = list()
    resultset=list()
    dataset = list()
    dataset2 = list()
    skip = ['@RELATION', '@ATTRIBUTE', '@DATA']
    with open(filename1) as file:
        for line in file:
            if not any(x in line for x in skip):
                result = [x.strip() for x in line.split(',')]
                if len(result) > 1:
                    dataset1.append(result[:len(result) - 1])
                    resultset.append([result[len(result) - 1]])


    with open(filename2) as file:
        for line in file:
            if not any(x in line for x in skip):
                result = [x.strip() for x in line.split(',')]
                if len(result) > 1:
                    dataset2.append(result[:len(result) - 1])
    
    for i,j in zip(dataset1,dataset2):
        dataset.append(i+j)

    dataset = np.array(dataset,dtype=np.float)
    #sc = StandardScaler()
    #dataset = sc.fit_transform(dataset)

    ohe = OneHotEncoder()
    resultset = ohe.fit_transform(resultset).toarray()

    return [dataset,resultset]


if __name__ == '__main__':
    #learning_one_file(5,3,'IrisGeometicFeatures_TrainingSet.txt','IrisGeometicFeatures_TestingSet.txt')
    learning_one_file(9600, 3,'IrisTextureFeatures_TrainingSet.txt','IrisTextureFeatures_TestingSet.txt')
    #learning_two_files(9605, 3,'IrisGeometicFeatures_TrainingSet.txt','IrisGeometicFeatures_TestingSet.txt','IrisTextureFeatures_TrainingSet.txt','IrisTextureFeatures_TestingSet.txt')
    