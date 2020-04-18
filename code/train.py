import os
import sys
import pickle
import lstm
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import random


def get_filename(args, time: int, util_name=""):   
    print(args)
    time = datetime.fromtimestamp(int(time))
    filename = str(time.strftime(str(args["learning_rate"])+"lr_"+str(args["max_epochs"])+'epochs_%b-%d-%Y_%H-%M-%S'))
    if util_name != "":
        filename = util_name+"_"+filename
    return filename

def get_batches():
    # TO DO
    pass


if __name__ == "__main__":
    max_epochs = 20
    learning_rate = 0.0001
    train_pickle_path = "data_hawkes/train.pkl"
    train = pickle.load(open(train_pickle_path, "rb"), encoding="latin1")
    time = datetime.now().timestamp()
    saves_path = os.path.join("./saves/", 
        get_filename({"learning_rate": learning_rate, "max_epochs": max_epochs}, time)
    )
    Path(saves_path).mkdir(parents=True, exist_ok=True)
    
    K=5+2+1
    
    lstm = lstm.LSTMTagger(K, 32, K)
    epochs_per_save = 5
    # batch_size = 100
    train_size = len(train)
    criterion_1 = nn.CrossEntropyLoss(reduction='mean')
    criterion_2 = nn.MSELoss(reduction='mean')    
    bos_index = 5
    eos_index = 6
    random.shuffle(train)
    optim = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
    n_samples = 100
    print(f"No of samples: {n_samples}\nNo of epochs: {max_epochs}\nLearning rate: {learning_rate}")
    for epoch in range(max_epochs):
        epoch_loss = 0
        for sample_idx in range(n_samples):
            sample = train[sample_idx]
            bos_tensor = torch.zeros(1,1,K)
            bos_tensor[:,:,bos_index]=1
            true_outputs_events = torch.zeros(len(sample)+1, 1, dtype=torch.int64)
            true_outputs_time = torch.zeros(len(sample)+1, 1)
            input_tensors = torch.zeros(len(sample)+1, 1, K)
            input_tensors[0,:,:] = bos_tensor
            for idx,val in enumerate(sample):
                tensor = torch.zeros(1,1,K)
                tensor[:,:,val["type_event"]]=1
                tensor[:,:,-1] = val["time_since_last_event"]
                input_tensors[idx+1,:,:] = tensor
                true_outputs_events[idx,0] = torch.tensor(val["type_event"], dtype=torch.int64)
                true_outputs_time[idx,0] = val["time_since_last_event"]

            true_outputs_events[:,-1] = eos_index        
            no_of_timesteps = input_tensors.size()[0]
            total_loss = 0

            for idx in range(no_of_timesteps):     
                input_tensor = torch.unsqueeze(input_tensors[idx,:,:], 0)                
                out = lstm(input_tensor)
                temp = torch.squeeze(out[:,:,:-1], dim=0)
                temp2 = true_outputs_events[idx,:]
                categorical_loss = criterion_1(temp, temp2)
                mse_loss = torch.sqrt(criterion_2(out[:,:,-1], true_outputs_time[idx,:]))
                loss = categorical_loss+mse_loss
                total_loss += loss
            # print(f"Sample {sample_idx}, Loss: {total_loss}")   
            epoch_loss += total_loss     
            total_loss.backward()
            optim.step()
        epoch_loss /= n_samples
        if epoch % epochs_per_save == epochs_per_save-1:
            filename = os.path.join(saves_path, "model_"+str(epoch+1)+"_epochs")

            torch.save(lstm, filename)
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")

