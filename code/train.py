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

def get_batches(samples, true_outputs_events, true_outputs_time, batch_size, padding_index, K):
    # TO DO
    pass    
    batches = []
    count = 0
    batch = []    
    batch_true_outputs_events = []
    batch_true_outputs_time = []
    padding_tensor = torch.zeros(1, 1, K)
    for idx, s in enumerate(samples):               
        batch.append(s)
        batch_true_outputs_events.append(true_outputs_events[idx])
        batch_true_outputs_time.append(true_outputs_time[idx])
        if (idx+1)%batch_size==0:
            max_len = max([val.size()[0] for val in batch])
            batch_new = []
            batch_true_outputs_events_new = []
            batch_true_outputs_time_new = []
            for val1,val2,val3 in zip(batch, batch_true_outputs_events, batch_true_outputs_time):
                if val1.size()[0] < max_len:                    
                    curr_len = val1.size()[0]
                    val1 = torch.cat((val1, padding_tensor.repeat(max_len-curr_len,1,1)), dim=0)
                    val2 = torch.cat((val2, torch.ones(1,1,dtype=torch.long).repeat(max_len-curr_len,1)), dim=0)
                    val3 = torch.cat((val3, torch.ones(1,1).repeat(max_len-curr_len,1)), dim=0)
                batch_new.append(val1)
                batch_true_outputs_events_new.append(val2)
                batch_true_outputs_time_new.append(val3)

            batches.append({
                "inputs": batch_new,
                "true_outputs_event": batch_true_outputs_events_new,
                "true_outputs_time": batch_true_outputs_time_new
            })       
                 
            batch = []
            batch_true_outputs_events = []
            batch_true_outputs_time = []

    if len(batch) > 0:
        max_len = max([val.size()[0] for val in batch])
        batch_new = []
        batch_true_outputs_events_new = []
        batch_true_outputs_time_new = []
        for val1,val2,val3 in zip(batch, batch_true_outputs_events, batch_true_outputs_time):
            if val1.size()[0] < max_len:                    
                curr_len = val1.size()[0]
                val1 = torch.cat((val1, padding_tensor.repeat(max_len-curr_len,1,1)), dim=0)
                val2 = torch.cat((val2, torch.ones(1,1,dtype=torch.long).repeat(max_len-curr_len,1)), dim=0)
                val3 = torch.cat((val3, torch.ones(1,1).repeat(max_len-curr_len,1)), dim=0)
            batch_new.append(val1)
            batch_true_outputs_events_new.append(val2)
            batch_true_outputs_time_new.append(val3)

        batches.append({
            "inputs": batch_new,
            "true_outputs_event": batch_true_outputs_events_new,
            "true_outputs_time": batch_true_outputs_time_new
        })                   
    return batches

def get_tensors(samples, K, bos_index, eos_index):
    samples_tensor_list = []
    true_outputs_events = []
    true_outputs_time = []
    
    for s in samples:
        bos_tensor = torch.zeros(1,1,K)
        bos_tensor[:,:,bos_index]=1
        true_outputs_e = torch.zeros(len(s)+1, 1, dtype=torch.int64)
        true_outputs_t = torch.zeros(len(s)+1, 1)
        input_tensors = torch.zeros(len(s)+1, 1, K)
        input_tensors[0,:,:] = bos_tensor

        for idx,val in enumerate(s):
            tensor = torch.zeros(1,1,K)
            tensor[:,:,val["type_event"]]=1
            tensor[:,:,-1] = val["time_since_last_event"]
            input_tensors[idx+1,:,:] = tensor
            true_outputs_e[idx,0] = torch.tensor(val["type_event"], dtype=torch.int64)
            true_outputs_t[idx,0] = val["time_since_last_event"]

        true_outputs_e[:,-1] = eos_index 
        samples_tensor_list.append(input_tensors)
        true_outputs_events.append(true_outputs_e)
        true_outputs_time.append(true_outputs_t)

    return samples_tensor_list, true_outputs_events, true_outputs_time


if __name__ == "__main__":
    max_epochs = 20
    learning_rate = 0.0001
    train_pickle_path = "data_hawkes/train.pkl"
    dev_pickle_path = "data_hawkes/dev.pkl"
    
    # 5 for event types, 3 for BOS, EOS and PADDING, 1 for regression value (delta t)
    K=5+3+1 

    epochs_per_save = 5
    batch_size = 8
    bos_index = 5
    eos_index = 6
    padding_index = 7
    time = datetime.now().timestamp()
    n_samples = 100
    seed_val = 23

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    train_samples = pickle.load(open(train_pickle_path, "rb"), encoding="latin1")
    dev_samples = pickle.load(open(train_pickle_path, "rb"), encoding="latin1")
    train_samples = [train_samples[idx] for idx in np.random.choice(np.arange(len(train_samples)), size=n_samples)]

    saves_path = os.path.join("./saves/", 
        get_filename({"learning_rate": learning_rate, "max_epochs": max_epochs}, time)
    )
    Path(saves_path).mkdir(parents=True, exist_ok=True)
    
    lstm = lstm.LSTMTagger(K, 32, K)
    train_size = len(train_samples)
    train_samples = sorted(train_samples, key=len)
    train_inputs, train_true_outputs_events, train_true_outputs_time = get_tensors(train_samples, K, bos_index, eos_index)
    train_batches = get_batches(train_inputs, train_true_outputs_events, train_true_outputs_time, batch_size, padding_index, K)

    criterion_1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_index)
    criterion_2 = nn.MSELoss(reduction='mean')    


    optim = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
    print(f"No of samples: {n_samples}\nNo of epochs: {max_epochs}\nLearning rate: {learning_rate}")
    for epoch in range(max_epochs):
        epoch_loss = 0
        for batch in train_batches:
            train_sample = torch.cat(batch["inputs"], dim=1)
            true_events = torch.cat(batch["true_outputs_event"], dim=1)
            true_delta_t = torch.cat(batch["true_outputs_time"], dim=1)
            no_of_timesteps = train_sample.size()[0]
            total_loss = 0
            for t in range(no_of_timesteps):
                out = lstm(torch.unsqueeze(train_sample[t,:,:], dim=0))

                categorical_loss = criterion_1(torch.squeeze(out[:,:,:-1], dim=0), true_events[t,:])
                mse_loss = torch.sqrt(criterion_2(out[:,:,-1], true_delta_t[t,:]))
                
                loss = categorical_loss+mse_loss
                total_loss += loss
            epoch_loss += total_loss     
            total_loss.backward()
            optim.step()
            
        epoch_loss /= n_samples
        if epoch % epochs_per_save == epochs_per_save-1:
            filename = os.path.join(saves_path, "model_"+str(epoch+1)+"_epochs")

            torch.save(lstm, filename)
        print(f"Epoch: {epoch+1}, Loss: {epoch_loss}")

