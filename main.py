
# coding: utf-8

# In[53]:


import pandas as pd
import random as rd
import numpy as np
import math
import sys
import os
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
from random import random


# In[54]:


C_Lambda = 0.9
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 200
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 2 #second dense layer would be the output
IsSynthetic = False
PHI = []


# ## Data preprocessing

# In[55]:


#Read the data from same_pairs.csv and diff_pairs.csv and merge them after sampling random points from diff_pairs.csv to reduce length
def readCsvAndConcatData(filename1, filename2, filename3):
    same_df = pd.read_csv(filename1)
    s = len(same_df)
    n = sum(1 for line in open(filename2))
    print(n)
    skip = sorted(rd.sample(range(1,n),(n-(s+1)))) #the 0-indexed header will not be included in the skip list
    #print(skip)
    diff_df = pd.read_csv(filename2,skiprows=skip)
    #print("Length of the same_df is", len(same_df))
    #print("Length of the diff_df is", len(diff_df))
    
    concat_df = pd.concat([same_df,diff_df])

    #print(concat_df)
    #print(len(same_df.columns))
    #print(len(diff_df.columns))
    
    features_df = pd.read_csv(filename3)
    features_df= features_df.iloc[:,1:]
    #features_df = features_df.iloc[:,1:]
    #print(features_df_new)
    new_df = concat_df.join(features_df.set_index([ 'img_id' ], verify_integrity=True), on=[ 'img_id_A' ], how='left')
    new_df = new_df.join(features_df.set_index([ 'img_id' ], verify_integrity=True) , on=[ 'img_id_B' ], how='left',lsuffix='_left', rsuffix='_right')
    #print(new_df)
    #print("Successfully concatenated the file fetching its features")
    return new_df

#GSC data has one less column and more number of rows thus new function defined to accomodate these changes in the csv.
def readCsvAndConcatDataGSC(filename1, filename2, filename3):
    same_df = pd.read_csv(filename1)
    s = len(same_df)
    n = sum(1 for line in open(filename2))
    #print(n)
    skip = sorted(rd.sample(range(1,n),(n-(s+1)))) #the 0-indexed header will not be included in the skip list
    #print(skip)
    diff_df = pd.read_csv(filename2,skiprows=skip)
    #print("Length of the same_df is", len(same_df))
    #print("Length of the diff_df is", len(diff_df))
    
    concat_df = pd.concat([same_df,diff_df])

    #print(concat_df)
    #print(len(same_df.columns))
    #print(len(diff_df.columns))
    
    features_df_new = pd.read_csv(filename3)
    features_df_new2 = pd.read_csv(filename3)
    #features_df_new = features_df.iloc[:,1:]
    #features_df_new2 = features_df.iloc[:,1:]
    #print(features_df_new)
    new_df = concat_df.join(features_df_new.set_index([ 'img_id' ], verify_integrity=True), on=[ 'img_id_A' ], how='left')
    new_df = new_df.join(features_df_new2.set_index([ 'img_id' ], verify_integrity=True) , on=[ 'img_id_B' ], how='left',lsuffix='_left', rsuffix='_right')
    #print(new_df)
    #print("Successfully concatenated the file fetching its features")
    return new_df

#Training target would be 80% of the total data
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

#Training data would be 80% of the total data
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

#Validation data would be 10% of the total data
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

#Validation target vector would be 10% of the entire target vector
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    print (str(ValPercent) + "% Val Target Data Generated..")
    return t


# ## Generate Datasets

# In[56]:


#Function for passing the human observed data for concatenation
def CreateHumanObservedDataConcat():
    #tableDataConcat = readCsvAndConcatData(os.path.join("Human_Observed_Data","same_pairs.csv"), os.path.join("Human_Observed_Data","diffn_pairs.csv"), os.path.join("Human_Observed_Data","HumanObserved-Features-Data.csv"))
    tableDataConcat = readCsvAndConcatData("./Human_Observed_Data/same_pairs.csv", "./Human_Observed_Data/diffn_pairs.csv", "./Human_Observed_Data/HumanObserved-Features-Data.csv")
    #tableDataConcat = readCsvAndConcatData("same_pairs.csv", "diffn_pairs.csv", "HumanObserved-Features-Data.csv")
    #print("====================================The appended data is given by===============================================\n")
    #print(tableDataConcat.loc[[1]])
    return tableDataConcat

#Function for passing the human observed data for subtraction
def CreateHumanObservedDataSub(tableDataConcat):
    tableDataF1List = tableDataConcat.iloc[:,3:12]
    #print("=============F1 LIST==============")
    #print(tableDataF1List)
    tableDataF2List = tableDataConcat.iloc[:,12:22]
    #print("=============F2 LIST==============")
    #print(tableDataF2List)
    tableDataSub = tableDataF1List.sub(tableDataF2List.values)
    tableDataSub['target']= tableDataConcat['target']
    #print("=====================================The subtracted data is given by============================================")
    #print(tableDataSub)
    return tableDataSub
    
#Function for passing the GSC data for concatenation    
def CreateGSCDataConcat():
    tableDataConcat = readCsvAndConcatDataGSC("./GSC_Data/same_pairs.csv", "./GSC_Data/diffn_pairs.csv","./GSC_Data/GSC-Features.csv")
    return tableDataConcat

#Function for passing the GSC data for subtraction
def CreateGSCDataSub(tableDataConcat):
    tableDataF1List = tableDataConcat.iloc[:,3:515]
    #print("=============F1 LIST==============")
    #print(tableDataF1List)
    tableDataF2List = tableDataConcat.iloc[:,515:1027]
    #print("=============F2 LIST==============")
    #print(tableDataF2List)
    tableDataSub = abs(tableDataF1List.sub(tableDataF2List.values))
    tableDataSub['target']= tableDataConcat['target']
    #print("=====================================The subtracted data is given by============================================")
    return tableDataSub

#Model made for Neural Network
def get_model():
    
    # Why do we need a model? -> ML would require the construction of a neural network, which has an input layer,
    # one or more hidden layers and an output layer. By choosing appropriate features for this model, we would be able to 
    # build its decision making capability
    # Why use Dense layer and then activation? -> Usage of dense layer suggests that each input is connected to each output within
    # its layer. Since FizzBuzz is a classification problem, such a connectivity is required for us. 
    # Why use sequential model with layers? -> Sequential model is used since for every input there is exactly one possible output.
    # The layers have been added (you would know why when you read about relu and softmax)
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size )) #Since this is the layer which is directly connected to the
    # input layer, the input size needs to be specified. For subsequent layers, it will take this value from the output of the 
    # previous layer
    #model.add(Activation('relu'))
    
    model.add(Activation('sigmoid'))
    
    # Why dropout?
    model.add(Dropout(drop_out)) #Dropout is a technique to reduce the dependency of the model on a particular neuron,
    # by dropping out random neuron values during the training. It is a model de-sensitizing technique
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax? -> Softmax is used for classification problems since it associates a probability value with each of the output
    # possibilities. The output with the highest probability would be chosen as the final output.
    
    model.summary()
    
    # Why use categorical_crossentropy? -> Categorical crossentropy would be used as the cost function, in order to estimate how
    # FAR we are from the actual result. If the predicted probability is closer to the actual probability, then the value of the 
    # cost will go on decreasing
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# ## Taking in User Options

# In[57]:


user_option = input("Type in 1 for Human Observed Dataset or 2 for GSC Dataset")

if user_option == "1":
    user_option_2 = input("Type in 1 for concatenated data and 2 for subtracted data")
    print("===========================================HumanObservedDataConcat=============================================")
    HumanObservedDataConcat = shuffle(CreateHumanObservedDataConcat())
    if user_option_2 == "1":
        HumanObservedDataConcat = HumanObservedDataConcat.loc[:, (HumanObservedDataConcat != 0).any(axis=0)]
        DataTarget = np.array(HumanObservedDataConcat.iloc[:,2]).flatten()
        DataNP = np.array(HumanObservedDataConcat.iloc[:,3:])
        print(DataTarget.shape)
        print(DataNP.shape)

    if user_option_2 == "2":
        print("===========================================HumanObservedDataSub=============================================")
        HumanObservedDataSub = shuffle(CreateHumanObservedDataSub(HumanObservedDataConcat))
        HumanObservedDataSub = HumanObservedDataSub.loc[:, (HumanObservedDataSub != 0).any(axis=0)]
        DataTarget = np.array(HumanObservedDataSub.iloc[:,len(HumanObservedDataSub.columns) - 1]).flatten()
        DataNP = np.array(HumanObservedDataSub.iloc[:,:len(HumanObservedDataSub.columns) - 1])
        print(DataTarget.shape)
        print(DataNP.shape)
        M = len(DataNP.shape)

elif user_option == "2":
    user_option_2 = input("Type in 1 for concatenated data and 2 for subtracted data")
    print("=============================================GSCDataConcat=========================================================")
    GSCDataConcat = shuffle(CreateGSCDataConcat())

    if user_option_2 == "1":
        GSCDataConcat = GSCDataConcat.loc[:, (GSCDataConcat != 0).any(axis=0)]
        DataTarget = np.array(GSCDataConcat.iloc[:,2]).flatten()
        DataNP = np.array(GSCDataConcat.iloc[:,3:])
        print(DataTarget.shape)
        print(DataNP.shape)
    if user_option_2 == "2":
        print("=============================================GSCDataSub============================================")
        GSCDataSub = shuffle(CreateGSCDataSub(GSCDataConcat))
        print(GSCDataSub.shape)
        #Eliminate the features that have value 0 for all data points
        GSCDataSub=GSCDataSub.loc[:, (GSCDataSub != 0).any(axis=0)]
        print(GSCDataSub.shape)
        DataTarget = np.array(GSCDataSub.iloc[:,len(GSCDataSub.columns) - 1 ]).flatten()
        DataNP = np.array(GSCDataSub.iloc[:,:len(GSCDataSub.columns) - 1 ])
        print(DataTarget.shape)
        print(DataNP.shape)

DataNP = np.transpose(DataNP)

user_option_3 = input("Type in 1 for Linear regression and 2 for Logistic regression and 3 for Neural Network")

if user_option_3 == "2":
    M = len(DataNP)

elif user_option_3 == "3":
    input_size = len(DataNP)


# ## Prepare Training Data

# In[58]:


#The training target would be 80% of the total given data
TrainingTarget = np.array(GenerateTrainingTarget(DataTarget,TrainingPercent))
#The training data would be 80% of the entire training data
TrainingData   = GenerateTrainingDataMatrix(DataNP,TrainingPercent)
#shape function in numpy used to return the shape of the numpy array
print(TrainingTarget.shape)
print(TrainingData.shape)


# ## Prepare Validation Data

# In[59]:


#The validation target would be 10% of the total given target
ValDataAct = np.array(GenerateValTargetVector(DataTarget,ValidationPercent, (len(TrainingTarget))))
#The validation data would be 10% of the total given data
ValData    = GenerateValData(DataNP,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)


# ## Prepare Test Data

# In[60]:


#The test target would be 10% of the total given target
TestDataAct = np.array(GenerateValTargetVector(DataTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
#The test data would be 10% of the total given data
TestData = GenerateValData(DataNP,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)


# In[61]:


def GenerateBigSigma(Data, MuMatrix,TrainingPercent):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])
        #varVect will contain the co-variance value for each data point (sigma squarred)
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
    #Converting the co-variance value to a matrix of BigSigma values that would be used in the gaussian radial basis function
    #formula
        BigSigma[j][j] = varVect[j]
  
    #Multiply the bigsigma value(variance) with a constant of 200 which will cause the points to be more spread out, but not
    #cause any difference to the end result
    BigSigma = np.dot(200,BigSigma)
    print ("BigSigma Generated..")
    print ("==============================================BIG SIGMA===========================================================")
    print(BigSigma)
    return BigSigma


def GetScalar(DataRow,MuRow, BigSigInv):
    #DataRow : 1x41 MuRow:1x41 which will make R:1x41 as well
    R = np.subtract(DataRow,MuRow)
    #BigSigInv: 41x41 np.transpose(R): 41x1 which will make T: 41x1
    T = np.dot(BigSigInv,np.transpose(R))
    #Since we need the squarred values, multiply again. R: 1x41 T:41x1 which will result in a scalar value
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    # equivalent to one term in the matrix PHI[R][C]
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))  
    #making a nested array with values as all zeroes, having dimension of 10(number of features) x training percent of the data
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    #The BigSig value is inversed since it has to be brought to the numerator so that it can be multiplied with the other matrices
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)): #len(MuMatrix) will return the number of rows that are present in this matrix i.e. 10
        for R in range(0,int(TrainingLen)):
            #For the entire training length, (0.8 * 69623) the value will be subtracted from centroid value given by the MuMatrix
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTestLinear(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    Y = 1/(1+ np.exp(-Y))
    return Y

def GetErmsLinear(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        #Getting the sum of squarred error values
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    #getting the percent accuracy
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    #Returns an appended string containing the accuracy and Erms error values
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def encodeLabel(TrainingTarget):
    processedLabel = []
    for trainingValue in TrainingTarget:
        if(trainingValue == 0):
            processedLabel.append([0])
        elif(trainingValue == 1):
            processedLabel.append([1])
    return np_utils.to_categorical(np.array(processedLabel),2)


# In[62]:


if user_option_3 == "1" or user_option_3 == "2":
    #Moore penrose pseudo inverse is very close to being the inverse of the matrix, with lesser computation time. Hence this is the
    #choice of inversion function that is used here
    ErmsArr = []
    AccuracyArr = []

    #kmeans is a clustering algorithm which is used in order to cluster the points that are closer together into groups.
    #This function is a part of the sklearn.cluster provider, the parameters given are the number of clusters (10) and the random
    #option will choose the centroids in the beginning at random. It performs 300 iterations for clustering by default, through which
    #it decides which cluster each data point should reside in. The data that is passed is the training data through the fit function
    #which tells us that using the given k-means configuration, fit the TrainingData into it.
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))

    #Mu is a 10 x 41 array which will contain the centroids for each of the 41 features over the entire data set.
    Mu = kmeans.cluster_centers_
    print(Mu)
    BigSigma     = GenerateBigSigma(DataNP, Mu, TrainingPercent)
    TRAINING_PHI = GetPhiMatrix(DataNP, Mu, BigSigma, TrainingPercent)
    #Weights calculated only on the basis of the training data
    W            = np.random.rand(M)
    print("++++++++++++++++++++++++++++++++++++Printing W++++++++++++++++++++++++++++++++++++")
    print(W)
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)


# In[63]:


if user_option_3 == "1" or user_option_3 == "2":
    print(Mu.shape)
    print(BigSigma.shape)
    print(TRAINING_PHI.shape)
    print(W.shape)
    print(VAL_PHI.shape)
    print(TEST_PHI.shape)


# In[64]:


La           = 2
learningRate = 0.015
L_Accuracy_Val   = []
L_Accuracy_TR    = []
L_Accuracy_Test  = []
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

if user_option_3 == "1" or user_option_3 == "2":
    W_Now        = np.dot(100, W)
    for i in range(0,400):
        #print ('---------Iteration: ' + str(i) + '--------------')
        if user_option_3 == "1":
            Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        elif user_option_3 == "2":
            TrainingData1 = np.transpose(TrainingData)
            Delta_E_D     = -np.dot((TrainingTarget[i] - (1 / (1 + np.exp(-(np.dot(np.transpose(W_Now),np.transpose(TrainingData1[i]))))))),TrainingData1[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next

        if user_option_3 == "1":
            #-----------------TrainingData Accuracy and Erms---------------------#
            TR_TEST_OUT   = GetValTestLinear(TRAINING_PHI,W_T_Next) 
            Erms_TR       = GetErmsLinear(TR_TEST_OUT,TrainingTarget)
            L_Accuracy_TR.append(float(Erms_TR.split(',')[0]))
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))

            #-----------------ValidationData Accuracy and Erms---------------------#
            VAL_TEST_OUT  = GetValTestLinear(VAL_PHI,W_T_Next) 
            Erms_Val      = GetErmsLinear(VAL_TEST_OUT,ValDataAct)
            L_Accuracy_Val.append(float(Erms_Val.split(',')[0]))
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))

            #-----------------TestingData Accuracy and Erms---------------------#
            TEST_OUT      = GetValTestLinear(TEST_PHI,W_T_Next) 
            Erms_Test = GetErmsLinear(TEST_OUT,TestDataAct)
            L_Accuracy_Test.append(float(Erms_Test.split(',')[0]))
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))

        elif user_option_3 == "2":
            #-----------------TrainingData Accuracy and Erms---------------------#
            #print(TrainingData1.shape)
            #print(W_T_Next.shape)
            TR_TEST_OUT   = GetValTestLinear(TrainingData1,W_T_Next) 
            Erms_TR       = GetErmsLinear(TR_TEST_OUT,TrainingTarget)
            L_Accuracy_TR.append(float(Erms_TR.split(',')[0]))
            L_Erms_TR.append(float(Erms_TR.split(',')[1]))

            #-----------------ValidationData Accuracy and Erms---------------------#
            VAL_TEST_OUT  = GetValTestLinear(np.transpose(ValData),W_T_Next) 
            Erms_Val      = GetErmsLinear(VAL_TEST_OUT,ValDataAct)
            L_Accuracy_Val.append(float(Erms_Val.split(',')[0]))
            L_Erms_Val.append(float(Erms_Val.split(',')[1]))

            #-----------------TestingData Accuracy and Erms---------------------#
            TEST_OUT      = GetValTestLinear(np.transpose(TestData),W_T_Next) 
            Erms_Test = GetErmsLinear(TEST_OUT,TestDataAct)
            L_Accuracy_Test.append(float(Erms_Test.split(',')[0]))
            L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        
if user_option_3 == "3":

    #print(len(TrainingTarget))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(len(processedLabel[1]))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    model = get_model()
    print("Prepared model")
    validation_data_split = 0.2
    num_epochs = 10000
    model_batch_size = 100
    tb_batch_size = 32 
    early_patience = 500 #tried values from 5-5000

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True) #Tensorboard is a tool that comes 
    # integrated with tensorflow which is used in order to visualize(In the form of graphs)
    # the changes that are taking place in the neural network over the
    # process of learning.
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min') #if the val_loss that is being
    # monitored becomes greater than the value that it was in the previous iteration, then the training would be stopped
    
    processedTrainingLabel= encodeLabel(TrainingTarget)
    print(len(processedTrainingLabel))
    #print(len(TrainingData))
    history = model.fit(np.transpose(TrainingData) # will tell you what signal (input data) we get
                        , processedTrainingLabel # The output that we are expecting
                        , validation_split=validation_data_split #a part of the training set will be used as the validation set
                        # in order to train the hyperparameters
                        , epochs=num_epochs #the number of times we go through the entire data set.
                        , batch_size=model_batch_size #the model will not take all the training values at once to train the system.
                        # it takes it in batches which is defined by the batch_size.
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )
    
    wrong   = 0
    right   = 0
    processedTestLabel = encodeLabel(TestDataAct)
    predictedTestLabel = []
    
    for i,j in zip(np.transpose(TestData),processedTestLabel):
        y = model.predict(np.array(i).reshape(-1,input_size))
        predictedTestLabel.append(y.argmax())

        if j.argmax() == y.argmax():
            right = right + 1
        else:
            wrong = wrong + 1
    print("Errors: " + str(wrong), " Correct :" + str(right))

    print("Testing Accuracy: " + str(right/(right+wrong)*100))


    #print("Model is ready")


# In[65]:


if user_option_3 == "1" or user_option_3 == "2":
    print ('----------Gradient Descent Solution--------------------')
    print ("M = 15 \nLambda  = 0.01\neta=0.01")
    print ("Accuracy Training   = " + str(np.around(max(L_Accuracy_TR),5)))
    print ("Accuracy Validation = " + str(np.around(max(L_Accuracy_Val),5)))
    print ("Accuracy Testing    = " + str(np.around(max(L_Accuracy_Test),5)))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


# # 
