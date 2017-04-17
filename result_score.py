# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:16:15 2017

@author: Administrator
"""
import pandas as pd
import numpy as np

#程序的数据文件
mainpath = r".\PredictResult.xlsx"

def countreallabel():#统计真实值结果
    labelset = []
    df = pd.read_excel(mainpath)
    for i in range(len(df)):
        temp = df.ix[i]
        labellist = temp[u"Reallabel"]
        labellist = labellist.strip()
        label = labellist.split(",")
        name = []
        for id in range(len(label)):
            name.append(int(label[id]))
        labelset.append(name)
    return labelset

def counttestlabel():#统计预测值结果
    labelset = []
    count = 0
    df = pd.read_excel(mainpath)
    for i in range(len(df)):
        temp = df.ix[i]
        labellist = temp[u"Predictlabel"]
        labellist = labellist.strip()
        label = labellist.split(",")
        name = []
        for id in range(len(label)):
            if label[id]!= '':
                name.append(int(label[id]))
            else:
                name.append(int(0))#存在输出为空的情况，将这种情况对应“0”
        labelset.append(name)
    return labelset
    
def CountRealresultandTestresult(maxlabel):#统计
    realresult = countreallabel()
    testresult = counttestlabel()

    #去除真实数据和测试数据中重复的标签
    for i in range(len(realresult)):
        name = sorted(list(set(testresult[i])))
        temp = sorted(list(set(realresult[i])))
        testresult[i] = name
        realresult[i] = temp
    return testresult, realresult

def Microaveraging(realresult,testresult):
    #比较realresult和testresult，并进行统计
    count = 0
    total = 0
    totaltest = 0
    for i in range(len(realresult)):
        for j in range(len(realresult[i])):
            total += 1
    
    for i in range(len(testresult)):
        for j in range(len(testresult[i])):
            totaltest += 1
    
    for i in range(len(testresult)):
        temp = testresult[i]
        for j in range(len(temp)):
            if temp[j] in realresult[i]:
                count += 1
            if j == 35:
                raise SystemExit
    sline = '*' * 45+"\n"
    sline2 = "\n"+'*' * 45 + "\n"
    with open(r".\ResultSocre.txt", "a") as f:
        f.write(sline)
        f.write("The number of correct:%s\n" % str(count))
        f.write("The number of allpredict:%s\n" % str(totaltest))
        f.write("The number of allsample:%s\n" % str(total))
        f.write(sline)
        f.write(sline2)
        MicroRecall = (float(count)/float(total))
        MicroPrecision = (float(count)/float(totaltest))
        MicroF = (2*MicroPrecision*MicroRecall)/(MicroPrecision+MicroRecall)
        f.write("The Microrecall:%.4f\n" % MicroRecall)
        f.write("The Microprecise:%.4f\n" % MicroPrecision)
        f.write("The MicroF:%.4f\n" % MicroF)
        f.write(sline)

def Macroaveraging(realresult, testresult):
    orderlabel = np.load(r'.\data\label.npy').tolist()
    lenlabel = len(orderlabel)
    testcorrectlabel =[0]*lenlabel#统计每类标签预测正确的个数
    testalllabel = [0]*lenlabel#统计每类标签预测的个数
    reallabel = [0]*lenlabel#统计真实测试集中每个类标签的个数
    row = len(testresult)
    for i in range(row):
        temp = testresult[i]
        for j in range(len(temp)):
            if temp[j]!=0:#0是没有预测出来的结果用0替代
                ix = orderlabel.index(temp[j])
                testalllabel[ix] += 1
                if temp[j] in realresult[i]:
                    index = orderlabel.index(temp[j])
                    testcorrectlabel[index] += 1

    #统计真实测试集中标签每个类标签的个数
    for i in range(len(realresult)):
        tmp = realresult[i]
        for j in range(len(tmp)):
            if tmp[j] in orderlabel:#真实测试集中存在不训练集中没有的数据则不统计该类
                ix = orderlabel.index(tmp[j])
                reallabel[ix] += 1

    Macro_Precision = 0
    Macro_Recall = 0
    sumprecision = 0
    for i in range(lenlabel):
        if(testalllabel[i]!= 0):#预测结果没有出现训练集中的标签 不作统计
            sumprecision += (float(testcorrectlabel[i])/float(testalllabel[i]))
    Macro_Precision = float(sumprecision)/float(lenlabel)#最后统计结果中是否考虑修改模型个数
    sumrecall = 0
    for i in range(lenlabel):
        if(reallabel[i]!=0):#实际测试集标签没有出现训练集中的标签 不作统计
            sumrecall += (float(testcorrectlabel[i]) / float(reallabel[i]))

    Macro_Recall = float(sumrecall)/float(lenlabel)
    Macro_F = (2*Macro_Precision*Macro_Recall)/(Macro_Precision+Macro_Recall)
    sline = "\n"+'*' * 45 + "\n"
    sline2 = '*' * 45 + "\n"
    with open(r".\ResultSocre.txt", "a") as f:
        f.write(sline)
        f.write("The Macrorecall:%.4f\n" % Macro_Recall)
        f.write("The Macroprecise:%.4f\n" % Macro_Precision)
        f.write("The MacroF:%.4f\n" % Macro_F)
        f.write(sline2)
    
def Hamming_Loss(realresult, testresult,maxlabel):
    #统计相关的标签没有出现在预测输出的标签集合中
    #不相关的标签出现在预测输出的标签集合中
    xorerror = 0
    for i in range(len(realresult)):
        temp1 = realresult[i]
        temp2 = testresult[i]
        for j in range(len(temp1)):
            if temp1[j] not in temp2:
                xorerror += 1
        for k in range(len(temp2)):
            if temp2[k] not in temp1:
                xorerror += 1

    hammingloss = xorerror/(float(len(realresult))*float(maxlabel))
    sline = "\n"+'*' * 45 + "\n"
    sline2 =  '*' * 45 + "\n"
    with open(r".\ResultSocre.txt", "a") as f:
        f.write(sline)
        f.write("The Hammingloss(the small the better):%.4f\n" % hammingloss)
        f.write(sline2)

#该评估标准用来表示预测的标签集合与真实的标签集合中相同的数据样本在测试数据集所占的比例大小
def Subset_accuracy(realresult,testresult):
    value = 0
    for i in range(len(realresult)):
        if realresult[i] == testresult[i]:
            value += 1
        else:
            pass
    subsetaccuracy = float(value)/float(len(realresult))
    sline = "\n"+'*' * 45 + "\n"
    sline2 = '*' * 45 + "\n"
    with open(r".\ResultSocre.txt", "a") as f:
        f.write(sline)
        f.write("The Subset accuracy:%.4f\n" % subsetaccuracy)
        f.write(sline2)

def Calculate_result(maxlabel):
    testresult,realresult = CountRealresultandTestresult(maxlabel)#统计预测结果和真实结果
    
    #基于类别的多标签分类评估标准
    Microaveraging(realresult, testresult)
    Macroaveraging(realresult, testresult)
    #基于样本的多标签分类评估标准
    Hamming_Loss(realresult, testresult,maxlabel)
    Subset_accuracy(realresult, testresult)

if __name__=="__main__":
     maxlabel = 23
     Calculate_result( maxlabel)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        