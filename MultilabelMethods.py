# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 10:54:00 2017

@author: lenovo
"""

import pandas as pd
import numpy as np
import jieba
import result_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


def OneHotLabel(datasource):
    labelnum = []
    df = pd.read_excel(datasource)
    for i in range(len(df)):
        temp = df.ix[i]
        label = temp[u"编号"]
        labellist = label.split(",")
        for j in range(len(labellist)):
            labelnum.append(int(labellist[j]))
    labelnum = sorted(set(labelnum))
    np.save(r'.\temp\label.npy',labelnum)
    row = len(df)
    cow = len(labelnum)
    data = np.zeros([row,cow])

    for i in range(len(df)):
        temp = df.ix[i]
        label = list(temp[u"编号"].split(","))
        result = np.zeros([23])
        for j in label:
            j = int(j)
            if j in labelnum:
                result[labelnum.index(j)] = 1
        data[i] = result
    np.save(r'.\temp\OneHotLabel.npy',data)
    
def SegtheSentence(datasource):
    fileSegWordDonePath =r'.\temp\resultofseg.txt'
    df = pd.read_excel(datasource)
    Allcomment = []
    for i in range(len(df)):
        temp = df.ix[i]
        Allcomment.append(temp[u"评论"])

    fileTrainSeg=[]
    for i in range(len(Allcomment)):
        fileTrainSeg.append([' '.join(list(jieba.cut(Allcomment[i], cut_all=False)))])

    with open(fileSegWordDonePath,'w',encoding="utf-8") as fW:
        for i in range(len(fileTrainSeg)):
            fW.write(fileTrainSeg[i][0]+"\n")
    return Allcomment

def ValuetheTermWeight():
    sentences = []
    with open(r'.\temp\resultofseg.txt',encoding="utf-8") as fileRaw:
        for line in fileRaw:
            sentences.append(line)
    
    count_Vector = CountVectorizer()#This class converts the words in the text into a word frequency matrix
    count = count_Vector.fit_transform(sentences)
    tf_transformer = TfidfTransformer(use_idf=False)#This class will count the TF-IDF weights of each word#use_idf=False
    tf_transformer.fit(count)
    
    data = tf_transformer.transform(count)
    data2 = data.todense()
    np.save(r'.\temp\Valueofsentence.npy',data2)


def My_train_test_split(X, Y, train_size):
    train_x = X[:int(len(X)*train_size)]
    test_x = X[int(len(X)*train_size):]
    train_y = Y[:int(len(Y) * train_size)]
    test_y = Y[int(len(Y) * train_size):]
    return train_x, train_y, test_x, test_y

    
def TrainMultilabelmodel(train_size):#should be seperated by training and predicting function
    X = np.load(r'.\temp\Valueofsentence.npy')
    Y = np.load(r'.\temp\OneHotLabel.npy')
    
    print(X.shape)
    print(Y.shape)
    X_train, Y_train, X_test, Y_test = My_train_test_split(X, Y, train_size)

    clf = GradientBoostingClassifier(n_estimators=55, learning_rate=0.1,subsample = 0.6,max_depth=10, random_state=0)
    
    clf2 = OneVsRestClassifier(clf)
    clf2.fit(X_train,Y_train)
    print('test acc:',clf2.score(X_test,Y_test))

    Y_predict = clf2.predict(X_test)
    np.save(r'.\temp\real_label.npy',Y_test)
    np.save(r'.\temp\test_label.npy',Y_predict)
    print( recall_score(Y_test,Y_predict,average='micro'))
    print( precision_score(Y_test,Y_predict,average='micro'))


def LabelCodeTrans(comment,test_rate,codepath):
    intlabel = []
    chineselabel = []
    # 读取码表的数据，将中英文进行对应
    with open(codepath, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            temp = i.split(" ")
            intlabel.append(int(temp[0].strip()))
            chineselabel.append(temp[1].strip())
    labeldict = {intlabel[i]: chineselabel[i] for i in range(len(intlabel))}

    content = comment[int(len(comment)*test_rate):]
    array1 = np.load(r'.\temp\real_label.npy')
    real_label,realchi_label = Onehot_CodeTable(array1,labeldict)

    # predict the results with numbers
    array2 = np.load(r'.\temp\test_label.npy')
    predict_label,predictchi_label = Onehot_CodeTable(array2,labeldict)
    data = pd.DataFrame(content, columns=['Comment'])
    data[u'Reallabel'] = real_label
    data[u'Predictlabel'] = predict_label
    data.to_excel('PredictResult.xlsx')

    #predict the results in Chinese
    data2 = pd.DataFrame(content, columns=['Comment'])
    data2[u'Reallabel'] = realchi_label
    data2[u'Predictlabel'] = predictchi_label
    data2.to_excel('PredictResultChinese.xlsx')


def Onehot_CodeTable(array,labeldict):
    labellist = np.load(r'.\temp\label.npy').tolist()
    global maxlabel
    maxlabel = len(labellist)
    label = array
    result = []
    ResultChinese = []
    a = 0
    for i in range(len(label)):
        temp = ''
        tempchinese = ''
        count = 0
        for index, value in enumerate(label[i]):
            if value == 1:
                count += 1
        countvalue = 0
        for index, value in enumerate(label[i]):
            if value == 1 and countvalue < count-1:
                countvalue += 1
                temp += str(labellist[index])+","
                tempchinese += labeldict.get(labellist[index])+","
            elif value == 1 and countvalue == count-1:
                countvalue += 1
                temp += str(labellist[index])
                tempchinese += labeldict.get(labellist[index])

        if(temp==""):
            a += 1
            result.append('0')#存在temp为空时，赋值为0
        else:
            result.append(temp)
        ResultChinese.append(tempchinese)
    print("The empty labelset")
    print(a)
    
    return result,ResultChinese

if __name__=="__main__":
    datasource = "./dataset/AllSampleOfJD01.xlsx"#输入训练和测试数据
    codepath = "./dataset/CodeOfJD01.csv"#输入对应的码表
    train_rate = 0.7
    OneHotLabel(datasource)
    comment = SegtheSentence(datasource)
    ValuetheTermWeight()
    TrainMultilabelmodel(train_rate)
    LabelCodeTrans(comment,train_rate,codepath)
    sline="\n\n\nThe GBDT,train_rate\n"
    with open(r".\ResultSocre.txt", "a") as f:
        f.write("\n\n\nThe GBDT,train_rate%f\n" % train_rate)
    result_score.Calculate_result(maxlabel)
