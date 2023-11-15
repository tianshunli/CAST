from ast import arg
from asyncore import write
import os
from tokenize import single_quoted
from unittest.mock import patch
from yaml import parse
import useful_tools
import time
import numpy as np
import argparse

import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.io as sio
import model
from torch import nn
import csv
from einops import rearrange, repeat
import csv
import json


parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, default='./config/train.json', help='json path')
args = parser.parse_args()

json_path = args.json_path
with open(json_path, 'r') as f:
    args = json.load(f)


trial_begin_time, begin_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime()), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
save_path = args["path"]["out_path"] + args["dataset"]["train"]["name"] + '/' +'output/' + trial_begin_time
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
if args["dataset"]["train"]["name"][0:5] == "river":
    head = 12
    dim_head = 33
elif args["dataset"]["train"]["name"][0:8] == "farmland":
    head = 10
    dim_head = 31
    
elif args["dataset"]["train"]["name"][0:3] == "USA": 
    head = 14
    dim_head = 22

# Download Data
if args["dataset"]["train"]["name"][0:8] == "farmland":
    input_file_name1 = './data/farmland/farm06.mat'
    input_file_name2 = './data/farmland/farm07.mat'
    input_label_name = './data/farmland/label.mat'
    farmland_before = sio.loadmat(input_file_name1)['imgh']
    farmland_after = sio.loadmat(input_file_name2)['imghl']
    farmland_label = sio.loadmat(input_label_name)['label']
    
elif args["dataset"]["train"]["name"][0:5] == "river":   
    input_file_name1 = './data/river/river_before.mat'
    input_file_name2 = './data/river/river_after.mat'
    input_label_name = './data/river/groundtruth.mat'
    farmland_before = sio.loadmat(input_file_name1)['river_before']#['imgh']
    farmland_after = sio.loadmat(input_file_name2)['river_after']   #['imghl']
    farmland_label = sio.loadmat(input_label_name)['lakelabel_v1']    #['label']
    for i in range(farmland_label.shape[0]):
        for j in range(farmland_label.shape[1]):
            if farmland_label[i][j] == 255:
                farmland_label[i][j] = 1

elif args["dataset"]["train"]["name"][0:3] == "USA":     
    input_file_name1 = './data/USA/USA_Change_Dataset.mat'
    farmland_before = sio.loadmat(input_file_name1)['T1']
    farmland_after = sio.loadmat(input_file_name1)['T2']   
    farmland_label = sio.loadmat(input_file_name1)['Binary']  
else:
    print("the data_name doesn't exist!")

dataset_shape_x, dataset_shape_y, dataset_bands_num = farmland_before.shape
CHANNELS = dataset_bands_num

farmland_before = np.float32(farmland_before)
farmland_after = np.float32(farmland_after)
farmland_before = farmland_before / farmland_before.max() 
farmland_after = farmland_after / farmland_after.max()
print(farmland_before.shape)
print(args["dataset"]["train"]["name"])



net = []
num_of_label = args["model"]["class_number"] 
if args["model"]["method"] == "CAST":  
    print("CAST")
    net = model.CAST(num_of_bands=dataset_bands_num, num_of_class=num_of_label, patch_size=args["dataset"]["train"]["patch_size"], 
                                         dataset=args["dataset"]["train"]["name"], head=head, dim_head=dim_head)
else:
    print("the network doesn't exist!")

train_turn = 0
average_OA = 0
average_KC = 0
average_F1 = 0
average_Pre = 0
average_Re = 0


# Load samples for training and testing
train_set = np.load("./data/"+args["dataset"]["train"]["name"]+"/train_set.npy",allow_pickle=True)
train_label = np.load("./data/"+args["dataset"]["train"]["name"]+"/train_lable.npy",allow_pickle=True)
test_sample_coordinare = np.load("./data/"+args["dataset"]["train"]["name"]+"/test_sample_coordinare.npy",allow_pickle=True)
test_sample_coordinare = test_sample_coordinare.tolist() 
train_sample_coordinate = np.load("./data/"+args["dataset"]["train"]["name"]+"/train_sample_coordinate.npy",allow_pickle=True)
train_sample_coordinate = train_sample_coordinate.tolist() 
before_after = np.load("./data/"+args["dataset"]["train"]["name"]+"/before_after.npy",allow_pickle=True)



for i in range(args["train_turn"]):
    epoch_a = args["dataset"]["train"]["epoch"]  
    train_set, train_label = torch.as_tensor(torch.from_numpy(train_set), dtype=torch.float32), torch.as_tensor(
        torch.from_numpy(train_label), dtype=torch.long)
    deal_dataset = torch.utils.data.TensorDataset(train_set, train_label)
    train_loader = torch.utils.data.DataLoader(deal_dataset, batch_size=args["dataset"]["train"]["batch_size"], shuffle=True, num_workers=0)
   
    os.environ["CUDA_VISIBLE_DEVICES"] = args["cuda"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net,device_ids = args["gpu_ids"])
    net = net.to(device)
    
    # Model parameter quantity
    num_parameters = sum([p.numel() for p in net.parameters() if p.requires_grad]) 
    print(f"total parameters: {num_parameters}")
    for n, p in net.named_parameters():
        if p.requires_grad:
            print(f"{n}, parameters: {p.numel()}")
    
    optimizer = torch.optim.Adadelta(net.parameters(), lr=args["dataset"]["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args["dataset"]["train"]["step_size"], gamma=0.1) 
    loss_func = torch.nn.CrossEntropyLoss() 

    time_train_begin = time.time()
    if args["phase"] == "train":
        loss_save = np.zeros(np.array(epoch_a))
        for epoch in range(epoch_a):
            net.train()
            scheduler.step()
            lr = scheduler.get_lr()
            print(epoch, scheduler.get_lr()[0])
            loss_list = np.zeros(np.array(train_set).shape[0])  
            loss_list_iter = 0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader): 
                inputs, labels = data
                inputs,labels = inputs.to(device),labels.to(device)  
                
                optimizer.zero_grad()
                outputs, x1_embedded, x1_decoder = net(inputs)   # inputs:[b, c*2, h, w]
               
                loss = loss_func(outputs, labels) 
                loss_list[loss_list_iter] = loss.cpu().detach().item()
                loss_list_iter += 1
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            avg_loss = np.average(loss_list)
           
            if args["verbose"]:
                print("epoch: " + str(epoch + 1) + "  loss: " + str(round(avg_loss, 6)) +
                        "  accuracy: " + str(round((correct / total) * 100, 4)) + " ")
            # save train_loss
            loss_save[epoch] = avg_loss
            np.save(save_path + '/' +"loss_save.npy",loss_save)
        torch.save(net.module, save_path +'/'+ str(epoch) + '_net.pkl')
        time_train_end = time.time()
        print("train end. time:" + str(round(time_train_end - time_train_begin)) + "s")
        torch.save(net.module, save_path +'/'+'net_model.pkl') 
        # Network training part end
        
    # test
    if args["phase"] == "test":
        net = torch.load(args["test"]["best_mode"])
    net = torch.load(save_path +'/'+'net_model.pkl')
    time_predict_begin = time.time()
    net = net.eval()
    predicted_total = np.zeros((dataset_shape_x, dataset_shape_y))

    for i in range(dataset_shape_x):
        test_set_row, test_label_row = useful_tools.get_set_row(before_after, farmland_label, i)  
        test_set_row, test_label_row = torch.as_tensor(torch.from_numpy(test_set_row), dtype=torch.float32), torch.as_tensor(
        torch.from_numpy(test_label_row), dtype=torch.long)
        test_dataset = torch.utils.data.TensorDataset(test_set_row, test_label_row)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=0)
    
        test_predicted = np.array([])
        for test_idx, test_data in enumerate(test_loader): 
            inputs, labels = test_data
            test_set_row,labels = inputs.to(device),labels.to(device)
            with torch.no_grad():
                outputs, x1_embedded, x1_decoder = net(test_set_row)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            test_predicted = np.append(test_predicted,predicted)

        predicted_total[i, :] = test_predicted

    acc_vector = useful_tools.calculate_acc(predicted_total, farmland_label,test_sample_coordinare)

    #Visualize the entire picture
    acc_vector_total = useful_tools.calculate_acc_total(predicted_total, farmland_label)
    predicted_color = acc_vector_total[-1]
    print("OA_total: " + str(round(acc_vector_total[0], 4)) + "  KC_total:" + str(round(acc_vector_total[1], 4)))
    print("PRE_total: " + str(round(acc_vector_total[2], 4)) + "  F1_total:" + str(round(acc_vector_total[3], 4)))
    
    
    average_OA += acc_vector[0]
    average_KC += acc_vector[1]
    average_F1 += acc_vector[3]
    average_Pre += acc_vector[2]
    average_Re += acc_vector[4]
    
    print("OA: " + str(round(acc_vector[0], 4)) + "  KC:" + str(round(acc_vector[1], 4)))
    print("PRE: " + str(round(acc_vector[2], 4)) + "  F1:" + str(round(acc_vector[3], 4)))
    time_predict_end = time.time()
    print("predict end. time:" + str(round(time_predict_end - time_predict_begin)) + "s")
    
    clist = [(0,"black"), (0.5,"Crimson"), (0.75, "Lime"),(1, "white")]  # Black, red, green, white
    #TN,FP,FN=2,TP=3
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
    matplotlib.image.imsave(save_path+"/" + 'prdict' + '.png',predicted_total,cmap=cmap)
    matplotlib.image.imsave(save_path+"/"+ 'gt' + '.png',farmland_label,cmap=cmap)
    matplotlib.image.imsave(save_path+"/" + 'predict_color' + '.png',predicted_color,cmap=cmap)


  
    with open("./save.csv","a+") as csvfile:
        writer = csv.writer(csvfile)  
        writer.writerows([["train_turn:",train_turn,"model: ",args["model"]["method"],
                           "OA:",str(round(acc_vector[0], 4)),
                           "KC:",str(round(acc_vector[1], 4)),"F1:",str(round(acc_vector[3], 4)),
                           "Pre: ",str(round(acc_vector[2], 4)),"Re: ",str(round(acc_vector[4], 4))]])
        writer.writerows([["TP: ",str(round(acc_vector[5])), "FN: ",str(round(acc_vector[6])),
                           "FP: ",str(round(acc_vector[7])), "TN: ",str(round(acc_vector[8]))]])
    train_turn += 1

average_OA = average_OA / args["train_turn"]
average_KC = average_KC / args["train_turn"]
average_F1 = average_F1 / args["train_turn"]
average_Pre = average_Pre / args["train_turn"]
average_Re = average_Re / args["train_turn"]

with open("./save.csv","a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[begin_time,
                           "average_OA:",str(round(average_OA,4)),"average_KC:",str(round(average_KC,4)),
                           "average_F1:",str(round(average_F1,4)),
                           "average_Pre:",str(round(average_Pre,4)),  
                           "average_Re: ",str(round(average_Re, 4))]])
        writer.writerows([["OA_total:",str(round(acc_vector_total[0], 4)),
                           "KC_total:",str(round(acc_vector_total[1], 4)),"F1_total:",str(round(acc_vector_total[3], 4)),
                           "Pre_total: ",str(round(acc_vector_total[2], 4)),"Re_total: ",str(round(acc_vector_total[4], 4))]])
        writer.writerows([["end: "]])
