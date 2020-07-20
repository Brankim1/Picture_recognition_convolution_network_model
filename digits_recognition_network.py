# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:13:03 2019

@author: Bran.Kim
"""
import tensorflow as tf  
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

def modle_neuron_network(X_train,X_test,y_train,y_test):
    model = tf.keras.models.Sequential()  #set model
    model.add(Flatten())  # input changed to one dimensional
    model.add(Dense(128, activation=tf.nn.relu))  # fully connected layer,，128 units， activation function is relu
    model.add(Dense(128, activation=tf.nn.relu)) 
    model.add(Dense(10, activation=tf.nn.softmax))  # output layer，10 units， use Softmax to get possibility
    
    model.compile(optimizer='adam',  # choose adam optimizer
                  loss='sparse_categorical_crossentropy',  # loss function that evaluates the error
                  metrics=['accuracy'])  # evaluation index
    model.fit(X_train, y_train, epochs=3)  # training model
    model.save('network_model.h5')#save model

def modle_convolution_network(X_train,X_test,y_train,y_test): 
    X_train=X_train.reshape(X_train.shape[0],8,8,1)   #reshape input dataset to fit model
    X_test=X_test.reshape(X_test.shape[0],8,8,1)
    
    model = tf.keras.models.Sequential()  
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(8, 8, 1))) #Convolution layer, kernel is 3*3
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2))) #Pooling layer
    model.add(Flatten())# use it to connecte Convolution layer and fully connected layer
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',  
                  loss='sparse_categorical_crossentropy',  
                  metrics=['accuracy'])  
    model.fit(X_train, y_train, epochs=3)  # train model, train 3 times
    model.save('convolution_network_model.h5') # save model

class selfknn():    # assigment 1, which get knn algorithm
   
    def knnalgorithm(self,X_test, X_train, y_train):
        #Euclidean Distance
        dataSetSize = X_train.shape[0]              #get trainset features length
        # use np.tile() for copy test features matrix up to trainset features length 
        diffMat = np.tile(X_test,(dataSetSize,1)) - X_train 
        sqDiffMat = diffMat ** 2 #square
        # add all elements to the left in row
        sqDistance = sqDiffMat.sum(axis=1)
        distance = sqDistance ** 0.5 #sqrt
        # use np.argsort() sorted index
        sortedIndex = distance.argsort()
        
        #find k NearestNeighbor
        classCount = {}  #create a list for label counts 
        for i in range(5):
            # use sortedIndex to get it labels
            voteLabel = y_train[sortedIndex[i]]
            # count the labels
            classCount[voteLabel] = classCount.get(voteLabel,0) + 1
        # sort the counts and return most labels
        sortedClassCount = sorted(classCount.keys(),reverse=True)
        return sortedClassCount[0]

def getAccuracy(dataset, predictions):# get accuracy in testing
    correct = 0
    for x in range(len(dataset)):
        if dataset[x] == predictions[x]:
            correct += 1
    return (correct / float(len(dataset)))

def cross_validation_sklearnknn(digits_X,digits_y): 
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.array_split(digits_X, 5)
    y_folds = np.array_split(digits_y, 5)
    y_test1=[]
    knn=KNeighborsClassifier()  # sklearn knn 
    predictions=[]
    for i in range(folds):   
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) #combine train dataset
        X_test =X_folds[i]   #combine test dataset
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])#get train label dataset
        y_test =y_folds[i]   #get test label dataset
        knn.fit(X_train,y_train)
        tem_predicts = knn.predict(X_test)
        y_test1.extend(y_test)  #test label Multiply in 5 folds
        predictions.extend(tem_predicts) #predict superposition in 5 folds
  
    predictions3=getAccuracy(y_test1,predictions)    #input all predict and true label
    print("*******************************************************\n")
    print("cross_validation_sklearnknn is done")
    print("cross_validation_sklearnknn Testing Score:",predictions3,"\n")
def cross_validation_selfknn(digits_X,digits_y):
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.array_split(digits_X, 5)
    y_folds = np.array_split(digits_y, 5)
    y_test1=[]
    clf=selfknn() #instantiation class selfknn
    predictions=[]
    for i in range(folds):
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:])
        X_test =X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_test =y_folds[i]

        for x in range(len(X_test)):    #import trainset for predict  and get trainset accuracy
            result=clf.knnalgorithm(X_test[x], X_train, y_train)  #get the x th trainset predict
            predictions.append(result)
        y_test1.extend(y_test)
    predictions3=getAccuracy(y_test1,predictions)    
    print("*******************************************************\n")
    print("cross_validation_selfknn is done")
    print("cross_validation_selfknn Testing Score:",predictions3,"\n")
def cross_validation_neuron_network(new_model,digits_X,digits_y):
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.array_split(digits_X, 5)
    y_folds = np.array_split(digits_y, 5)
    prediction1=[]
    y_test1=[]
    for i in range(folds):
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:])
        X_test =X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_test =y_folds[i]
        new_model.fit(X_train,y_train)          #mosel predict
        predictions = new_model.predict(X_test)
        for k in range(len(predictions)):
            tem_predicts=np.argmax(predictions[k])
            prediction1.append(tem_predicts)    #predict matrix superposition 
        y_test1.extend(y_test)#test label superposition
        
    predictions3=getAccuracy(y_test1,prediction1)    
    print("*******************************************************\n")
    print("cross_validation_neuron_network is done")
    print("cross_validation_neuron_network Testing Score:",predictions3,"\n")
def cross_validation_convolution_network(new_model,digits_X,digits_y):
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.array_split(digits_X, 5)
    y_folds = np.array_split(digits_y, 5)
    prediction1=[]
    y_test1=[]
    for i in range(folds):
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:])
        X_test =X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_test =y_folds[i]
        X_train=X_train.reshape(X_train.shape[0],8,8,1)
        X_test=X_test.reshape(X_test.shape[0],8,8,1)
        new_model.fit(X_train,y_train, epochs=3)
        predictions = new_model.predict(X_test)
        for k in range(len(predictions)):
            tem_predicts=np.argmax(predictions[k])
            prediction1.append(tem_predicts)
        y_test1.extend(y_test)
        
    predictions3=getAccuracy(y_test1,prediction1)    
    print("*******************************************************\n")   
    print("cross_validation_convolution_network is done")
    print("cross_validation_convolution_network Testing Score:",predictions3,"\n") 


def plot_confusion_matrix(y_test,predict):
    #get confusion matrix
    matrix=np.zeros((10, 10)) #set zero matrix to confusion matrix
    matrix=matrix.astype(int) #let zero matrix number bacome int
    count=0
    for i in range(10): #sum row
        for j in range(10): #sum column
            for k in range(len(y_test)):
                if y_test[k]==i  and predict[k]==j: 
                    count+=1
            matrix[i,j]=count
            count=0
    
    #print(materx)
    plt.matshow(matrix,cmap=plt.cm.Blues)
    plt.colorbar()# colour
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()

def confusion_matrix_sklearnknn(X_train,X_test,y_train,y_test):
    knn=KNeighborsClassifier()
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    plot_confusion_matrix(y_test,predict)

def confusion_matrix_selfknn(X_train,X_test,y_train,y_test):
    predict=[]
    clf=selfknn() #instantiation class selfknn
    for x in range(len(X_test)):    #import trainset for predict  and get trainset accuracy
        result=clf.knnalgorithm(X_test[x], X_train, y_train)  #get the x th trainset predict
        predict.append(result)
    plot_confusion_matrix(y_test,predict)

def confusion_matrix_neuron_network(new_model,X_train,X_test,y_train,y_test):
     new_model.fit(X_train,y_train)             #train model
     predictions = new_model.predict(X_test)
     predict=[]
     for k in range(len(predictions)):       #get predictions
        tem_predicts=np.argmax(predictions[k])
        predict.append(tem_predicts)
     plot_confusion_matrix(y_test,predict)

def confusion_matrix_convolution_network(new_model,X_train,X_test,y_train,y_test):
     X_train=X_train.reshape(X_train.shape[0],8,8,1)  #change input size
     X_test=X_test.reshape(X_test.shape[0],8,8,1)
     new_model.fit(X_train,y_train, epochs=3)      #train model
     predictions = new_model.predict(X_test)
     predict=[]
     for k in range(len(predictions)): #predict
         tem_predicts=np.argmax(predictions[k])
         predict.append(tem_predicts)
     plot_confusion_matrix(y_test,predict)


def roc_curve(predictions,y_test):
    for i in range(0,10):   #draw 0-9 different roc curve
        predict=[]
        predict=predictions[:,i]   #get predict confindence in a classes
        predict_array=np.array(predict).reshape(360,1)  #reshape it 
        y_test_array=y_test.reshape(360,1)              #reshapr it
        tem_array1=np.hstack((predict_array,y_test_array)).tolist()    #combine predict_array and y_test_array
        
        rocinputdata=sorted(tem_array1,key=lambda x:(x[0]),reverse=True)  # descending sort tem_array1 in predict_array
        
        tpr=[]
        fpr=[]
        fp=0
        tp=0
        count=0
        for k in range(len(rocinputdata)):# get count the y_test lable is true 
            if(rocinputdata[k][1]==i):
                    count+=1
        for k in range(len(rocinputdata)):# let  threshold value=predict possibility, so loop len(rocinputdata)
            for m in range(k):            
                if(rocinputdata[m][1]==i):
                    tp+=1
                elif(rocinputdata[m][1]!=i):
                    fp+=1
            
            fn=count-tp
            tn=360-count-fp
            tem_tpr=tp/(tp+fn)
            tem_fpr=fp/(fp+tn) 
            tpr.append(tem_tpr)
            fpr.append(tem_fpr)        
            fp=0
            tp=0       
        plt.plot(fpr,tpr, linewidth = 3)
        plt.plot([0,1],[0,1],'k--')
        plt.axis([0,1,0,1.05])
        plt.title("%s Receiver operating characteristic for"%(i))
        plt.xlabel("False Positive Rate")
        plt.ylabel('True Positive Rate')
        plt.show()

def roc_neuron_network(new_model,X_train,X_test,y_train,y_test):
    new_model.fit(X_train,y_train)
    predictions = new_model.predict(X_test)
    roc_curve(predictions,y_test)

def roc_convolution_network(new_model,X_train,X_test,y_train,y_test):
    X_train=X_train.reshape(X_train.shape[0],8,8,1)
    X_test=X_test.reshape(X_test.shape[0],8,8,1)
    new_model.fit(X_train,y_train, epochs=3)
    predictions = new_model.predict(X_test)
    roc_curve(predictions,y_test)

def main():
    digits=datasets.load_digits()# get dataset
    digits_X=digits.data
    digits_y=digits.target
    X_train,X_test,y_train,y_test=train_test_split(digits_X,digits_y,test_size=0.2,random_state=1) #split dataset
   
    #modle_neuron_network(X_train,X_test,y_train,y_test) #train neuron_network
    #modle_convolution_network(X_train,X_test,y_train,y_test)#train convolution_network
    print("*******************************************************\n"
          "Neuron_network modeling is done\n"
          "Convolution_network modeling is done\n")
    

    for y in range(1000000): # can loop execute many times
        print("*******************************************************")
        print("please choose model to operate\n"
          "1. SklearnKnn\n"
          "2. Selfknn\n"
          "3. Neuron network\n"
          "4. Convolution network\n"
          "5. Exit")
                                    
        index=int(input())
       
        if index==1:
            print("*******************************************************")
            print("sklearnKnn Please choose you want to show\n"
                  "1. Cross validation\n"
                  "2. Confusion matrix\n")
            index2=int(input())
            if(index2==1):
                cross_validation_sklearnknn(digits_X,digits_y)
                continue;
            if(index2==2):
                confusion_matrix_sklearnknn(X_train,X_test,y_train,y_test)
                continue;
            
        if index==2:
            print("*******************************************************")
            print("Selfknn Please choose you want to show\n"
                  "1. Cross validation\n"
                  "2. Confusion matrix\n")
            index3=int(input())
            if(index3==1):
                cross_validation_selfknn(digits_X,digits_y)
                continue;
            if(index3==2):
                confusion_matrix_selfknn(X_train,X_test,y_train,y_test)
                continue;
          
        if index==3:
            new_model = tf.keras.models.load_model('network_model.h5') #load network_model
            print("*******************************************************")
            print("Neuron network Please choose you want to show\n"
                  "1. Cross validation\n"
                  "2. Confusion matrix\n"
                  "3. Roc curve\n")
            index4=int(input())
            if(index4==1):
                cross_validation_neuron_network(new_model,digits_X,digits_y)
                continue;
            if(index4==2):
                confusion_matrix_neuron_network(new_model,X_train,X_test,y_train,y_test)
                continue;
            if(index4==3):
                roc_neuron_network(new_model,X_train,X_test,y_train,y_test)
                continue;
        
        if index==4:
            new_model = tf.keras.models.load_model('convolution_network_model.h5') #load convolution_network_model
            print("*******************************************************")
            print("Convolution network Please choose you want to show\n"
                  "1. Cross validation\n"
                  "2. Confusion matrix\n"
                  "3. Roc curve\n")
            index5=int(input())
            if(index5==1):
                cross_validation_convolution_network(new_model,digits_X,digits_y)
                continue;
            if(index5==2):
                confusion_matrix_convolution_network(new_model,X_train,X_test,y_train,y_test)
                continue;
            if(index5==3):
                roc_convolution_network(new_model,X_train,X_test,y_train,y_test)
                continue;
        if index==5:
            break;
if __name__=="__main__":
    main()