# from itertools import Predicate
import numpy as np
import copy
import pandas
from sklearn.metrics import precision_recall_curve
from torch import device



def get_set_row(dataset, ground_truth, index):
    patch_size = dataset.shape[0] - ground_truth.shape[0] + 1
    test_sample = list()
    test_label = list()
    for j in range(ground_truth.shape[1]):
        test_sample.append(dataset[index:index+patch_size, j:j+patch_size])
        test_label.append(ground_truth[index,j])
    test_sample, test_label = np.array(test_sample), np.array(test_label)
    test_sample = test_sample.transpose((0, 3, 1, 2))
    return test_sample, test_label


def calculate_acc(confusion_matrix, farmland_label, test_sample_coordinare):
    TN = 0  # The number of correct predictions has not changed
    FN = 0  # Changed, predicted to be 0
    FP = 0  # Unchanged, predicted to be 1
    TP = 0  # The change prediction is 1: the prediction is correct
    for i in range(len(test_sample_coordinare)):
        test1 = test_sample_coordinare[i]
        for j in range(len(test1)):
            
            if confusion_matrix[test1[j][0]][test1[j][1]] == farmland_label[test1[j][0]][test1[j][1]] and farmland_label[test1[j][0]][test1[j][1]] == 0:
                TN += 1
            elif confusion_matrix[test1[j][0]][test1[j][1]] != farmland_label[test1[j][0]][test1[j][1]] and farmland_label[test1[j][0]][test1[j][1]] == 0:
                FP += 1
            elif confusion_matrix[test1[j][0]][test1[j][1]] != farmland_label[test1[j][0]][test1[j][1]] and farmland_label[test1[j][0]][test1[j][1]] == 1:
                FN += 1
            elif confusion_matrix[test1[j][0]][test1[j][1]] == farmland_label[test1[j][0]][test1[j][1]] and farmland_label[test1[j][0]][test1[j][1]] == 1:
                TP += 1
    # Calculate OA and Kappa coefficiency
    # print("TP:"+ str(round(TP)),"FN:" + str(round(FN)))
    # print("FP:" + str(round(FP)), "TN:" + str(round(TN)))
    precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    F1_Score = (2*precision*Recall) / (precision+Recall)
    num_tatel = TN + TP + FP + FN
    OA = float((TN + TP)) / (num_tatel)
    PRE = float((TP + FP)*(TP + FN)) / (num_tatel ** 2) + float((FN + TN)*(FP + TN)) / (num_tatel ** 2)
    KC =(OA - PRE) / (1 - PRE)
    return OA, KC, precision ,F1_Score,Recall,TP,FN,FP,TN
    
#  The entire picture is used as a test to calculate the accuracy
def calculate_acc_total(confusion_matrix, farmland_label):
    TN = 0  
    FN = 0  
    FP = 0 
    TP = 0  
    h, w = farmland_label.shape
    predicted_color = np.zeros((h, w))
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
                                #  Predict 
                                # change       # unchange
            # label     change    TP            FN
            # label     unchange  FP            TN
            if confusion_matrix[i][j] == farmland_label[i][j] and farmland_label[i][j] == 0:
                TN += 1   #block
                predicted_color[i][j] = 0
            elif confusion_matrix[i][j] != farmland_label[i][j] and farmland_label[i][j] == 0:
                FP += 1     #red
                predicted_color[i][j] = 1
            elif confusion_matrix[i][j] != farmland_label[i][j] and farmland_label[i][j] == 1:
                FN += 1    #green
                predicted_color[i][j] = 3
            elif confusion_matrix[i][j] == farmland_label[i][j] and farmland_label[i][j] == 1:
                TP += 1   #white
                predicted_color[i][j] = 4
            else:
                pass
    # print("TP:"+ str(round(TP)),"FN:" + str(round(FN)))
    # print("FP:" + str(round(FP)), "TN:" + str(round(TN)))
    precision = TP / (TP+FP)
    Recall = TP / (TP+FN)
    F1_Score = (2*precision*Recall) / (precision+Recall)
    num_tatel = TN + TP + FP + FN
    OA = float((TN + TP)) / (num_tatel)
    PRE = float((TP + FP)*(TP + FN)) / (num_tatel ** 2) + float((FN + TN)*(FP + TN)) / (num_tatel ** 2)
    KC =(OA - PRE) / (1 - PRE)
    return OA, KC, precision ,F1_Score,Recall,TP,FN,FP,TN,predicted_color








