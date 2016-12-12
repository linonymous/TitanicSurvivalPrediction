# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 09:14:05 2016

@author: LINONYMOUS
"""
import csv
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# survived Pclass Sex Age Fare
# Loading The Data
#dataframe = pd.read_csv('train.csv',usecols=[2,4,5,9], engine='python')
#pre_data = dataframe.values
#dataframe = pd.read_csv('train.csv',usecols=[1], engine='python')
#target = dataframe.values
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
null = ''
data=[]
target=[]
dic = {'male': 0,'female' : 1}
with open('train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    tupl=[]
    for row in reader:
        if row['Survived']!=null and row['Pclass']!=null and row['Sex']!=null and row['Age']!=null and row['Fare']!=null:
            tupl=[]
            tupl.append(row['Pclass'])
            tupl.append(dic[str(row['Sex'])])
            tupl.append(row['Age'])
            tupl.append(row['Fare'])
            data.append(tupl)
            target.append(row['Survived'])
            
"""
dataframe = pd.read_csv('test.csv',usecols=[2,4,5,9], engine='python')
testX = dataframe.values
dataframe = pd.read_csv('test.csv',usecols=[1], engine='python')
testY = dataframe.values
"""


#print len(trainX)
"""
for row in pre_data[:]:
    row[1]=dic[str(row[1])]
"""
#pre_data.astype('float32')
#target.astype('float32')
"""
data=[]
for row in pre_data:
    if not row[0] and not row[1] and not row[2] and not row[3]:# and row[2]!='' and row[3]!='':
        data.append(row)
"""    

data = np.array(data)
target = np.array(target)
trainX = data[:-90]
trainY = target[:-90]
testX = data[-90:]
testY = target[-90:]

model = Sequential()
model.add(Dense(12,input_dim=4,init = 'uniform',activation='relu'))
model.add(Dense(10,init = 'uniform',activation='relu'))
model.add(Dense(8,init = 'uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(trainX,trainY,nb_epoch=100,batch_size=10,verbose=1)

prediction = model.predict(testX)

#print metrics.accuracy_score(testY,prediction)*100
scores = model.evaluate(testX,testY)

print ('Accuracy : %.2f') % (scores[1]*100)
print len(data)