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
import pickle
import logging




def get_device(device_no: int, logger):
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:"+str(device_no))
        logger.info('There are %d GPU(s) available.' + str(torch.cuda.device_count()))
        logger.info('We will use the GPU:'+ str(torch.cuda.get_device_name(0)))
    # If not...
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def get_filename(args: dict, time: int, util_name=""):   
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
                    val2 = torch.cat((val2, torch.ones(1,1,dtype=torch.float).repeat(max_len-curr_len,1)), dim=0)
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
                val2 = torch.cat((val2, torch.zeros(1,1).repeat(max_len-curr_len,1)), dim=0)
                val3 = torch.cat((val3, torch.zeros(1,1).repeat(max_len-curr_len,1)), dim=0)
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
        true_outputs_e = torch.zeros(len(s)+1, 1)
        true_outputs_t = torch.zeros(len(s)+1, 1)
        input_tensors = torch.zeros(len(s)+1, 1, K)
        input_tensors[0,:,:] = bos_tensor

        for idx,val in enumerate(s):
            tensor = torch.zeros(1,1,K, dtype=torch.float64)
            tensor[:,:,val["type_event"]]=1
            tensor[:,:,-1] = torch.tensor(val["time_since_last_event"], dtype=torch.float64)
            input_tensors[idx+1,:,:] = tensor
            true_outputs_e[idx,0] = torch.tensor(val["type_event"], dtype=torch.long)
            true_outputs_t[idx,0] = torch.tensor(val["time_since_last_event"], dtype=torch.float64)

        true_outputs_e[-1,:] = eos_index 
        true_outputs_t[-1,:] = 0
        samples_tensor_list.append(input_tensors)
        true_outputs_events.append(true_outputs_e)
        true_outputs_time.append(true_outputs_t)

    return samples_tensor_list, true_outputs_events, true_outputs_time



def train_model(train_samples, dev_samples, learning_rate, max_epochs, batch_size, K, hidden_size, bos_index, eos_index, padding_index, dir_name):
    
    epochs_per_save = 5
    

    time = datetime.now().timestamp()
    #saves_path = os.path.join("./saves/" + str(datetime.fromtimestamp(int(time)).strftime(dir_name + '_%b-%d-%Y_%H-%M-%S')) + "/", 
    #    get_filename({"learning_rate": learning_rate, "max_epochs": max_epochs}, time)
    #)
    saves_path = "./saves/" + str(datetime.fromtimestamp(int(time)).strftime(dir_name + '_%b-%d-%Y_%H-%M-%S')) + "/"
    Path(saves_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(saves_path,'run.log'), filemode='w', format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("Training Data Directory: %s", dir_name)
    logger.info("Learning rate: " + str(learning_rate) + " Hidden size: " + str(hidden_size)+ " Batch size:" + str(batch_size))
    device = get_device(0, logger)

    mlstm = lstm.LSTMTagger(K, hidden_size, K)
    if torch.cuda.is_available():
        mlstm.cuda(device=device)

    criterion_1 = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_index)
    criterion_2 = nn.MSELoss(reduction='mean') 

    optim = torch.optim.Adam(mlstm.parameters(), lr=learning_rate)

    n_samples = len(train_samples)
    train_samples = [train_samples[idx] for idx in np.random.choice(np.arange(len(train_samples)), size=n_samples)]

    train_samples = sorted(train_samples, key=len)
    train_inputs, train_true_outputs_events, train_true_outputs_time = get_tensors(train_samples, K, bos_index, eos_index)
    train_batches = get_batches(train_inputs, train_true_outputs_events, train_true_outputs_time, batch_size, padding_index, K)

    dev_size = len(dev_samples)
    dev_samples = sorted(dev_samples, key=len)
    dev_inputs, dev_true_outputs_events, dev_true_outputs_time = get_tensors(dev_samples, K, bos_index, eos_index)
    dev_batches = get_batches(dev_inputs, dev_true_outputs_events, dev_true_outputs_time, batch_size, padding_index, K)

    logger.info(f"No of samples: {n_samples}, No of epochs: {max_epochs}")
    results = []
    for epoch in range(max_epochs):
        epoch_loss = 0
        event_losses = []
        time_losses = []
        for batch in train_batches:
            train_sample = torch.cat(batch["inputs"], dim=1).to(device)
            true_events = torch.cat(batch["true_outputs_event"], dim=1).to(device, dtype=torch.long)
            true_delta_t = torch.cat(batch["true_outputs_time"], dim=1).to(device)
            # train_sample = torch.cat(batch["inputs"], dim=1)
            # true_events = torch.cat(batch["true_outputs_event"], dim=1)
            # true_delta_t = torch.cat(batch["true_outputs_time"], dim=1)
            no_of_timesteps = train_sample.size()[0]
            total_loss = 0
            avg_time_loss = 0
            avg_event_loss = 0
            for t in range(no_of_timesteps):
                out = mlstm(torch.unsqueeze(train_sample[t,:,:], dim=0))
                temp=out[:,:,:-1]
                temp2 = true_events[t,:]
                temp3 = true_delta_t[t,:]
                categorical_loss = criterion_1(torch.squeeze(temp, dim=0), torch.squeeze(temp2, dim=0))
                mse_loss = torch.sqrt(criterion_2(out[:,:,-1], torch.unsqueeze(true_delta_t[t,:], dim=0)))
                avg_event_loss += categorical_loss
                avg_time_loss += mse_loss
                
                loss = categorical_loss+mse_loss
                total_loss += loss
            
            avg_time_loss /= no_of_timesteps
            avg_event_loss /= no_of_timesteps
            event_losses.append(avg_event_loss.detach())
            time_losses.append(avg_time_loss.detach())

            total_loss = total_loss/no_of_timesteps
            epoch_loss += total_loss     
            total_loss.backward()
            optim.step()
        
        epoch_loss /= n_samples

        dev_loss = 0
        event_losses_dev = []
        time_losess_dev = []
        for batch in dev_batches:
            dev_sample = torch.cat(batch["inputs"], dim=1).to(device)
            dev_true_events = torch.cat(batch["true_outputs_event"], dim=1).to(device, dtype=torch.long)
            dev_true_delta_t = torch.cat(batch["true_outputs_time"], dim=1).to(device)
            # train_sample = torch.cat(batch["inputs"], dim=1)
            # true_events = torch.cat(batch["true_outputs_event"], dim=1)
            # true_delta_t = torch.cat(batch["true_outputs_time"], dim=1)
            no_of_timesteps =dev_sample.size()[0]
            total_loss = 0
            avg_event_loss_dev = 0
            avg_time_loss_dev = 0
            for t in range(no_of_timesteps):
                out = mlstm(torch.unsqueeze(dev_sample[t,:,:], dim=0))
                temp = out[:,:,:-1]
                temp2 = dev_true_events[t,:]
                categorical_loss = criterion_1(torch.squeeze(temp, dim=0), torch.squeeze(temp2, dim=0))
                avg_event_loss_dev += categorical_loss

                mse_loss = torch.sqrt(criterion_2(out[:,:,-1], torch.unsqueeze(dev_true_delta_t[t,:], dim=0)))
                avg_time_loss_dev += mse_loss

                loss = categorical_loss+mse_loss
                total_loss += loss
            avg_event_loss_dev /= no_of_timesteps
            avg_time_loss_dev /= no_of_timesteps
            total_loss = total_loss/no_of_timesteps
            dev_loss += total_loss 
            
            event_losses_dev.append(avg_event_loss_dev.detach())
            time_losess_dev.append(avg_time_loss_dev.detach())

        dev_loss /= dev_size


        avg_event_loss_val = np.mean(avg_event_loss.detach().cpu().numpy())
        avg_time_loss_val = np.mean(avg_time_loss.detach().cpu().numpy())
        avg_event_loss_dev_val = np.mean(avg_event_loss_dev.detach().cpu().numpy())
        avg_time_loss_dev_val = np.mean(avg_time_loss_dev.detach().cpu().numpy())

        results.append({'epoch': epoch, 'train_loss': epoch_loss.item(), 'dev_loss': dev_loss.item(), 
                        'event_loss': avg_event_loss_val, 'time_loss': avg_time_loss_val,
                        'dev_event_loss':  avg_event_loss_dev_val, 'dev_time_loss':  avg_time_loss_dev_val})
        if epoch % epochs_per_save == epochs_per_save-1:
            filename = os.path.join(saves_path, "model_"+str(epoch+1)+"_epochs")
            torch.save(mlstm, filename)
            pickle.dump({'results': results, 
                        'args': {'learning_rate': learning_rate, 'max_epochs': max_epochs, 'batch_size': batch_size, 'K': K, 'hidden_size': hidden_size }}, 
                        open(saves_path+"/losses.pkl", 'wb'))

        
        logger.info("Epoch: " + str(epoch+1) + ", Train Loss: " + str(epoch_loss.item()) + ", Dev Loss: " + str(dev_loss.item()) +", Train Event loss: " + str(avg_event_loss_val) + 
                   ", Train Time loss " + str(avg_time_loss_val) + ", Dev Event loss " + str(avg_event_loss_dev_val) + ", Dev Time loss " + str(avg_time_loss_dev_val))
                        
    pickle.dump({'results': results, 
                 'args': {'learning_rate': learning_rate, 'max_epochs': max_epochs, 'batch_size': batch_size, 'K': K, 'hidden_size': hidden_size }}, 
                  open(saves_path+"/losses.pkl", 'wb'))
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    dir_name = "data_retweet"
    train_pickle_path = dir_name + "/train.pkl"
    dev_pickle_path = dir_name + "/dev.pkl"
    

    # logger.info("Training Data Directory: %s", dir_name)

    max_epochs = 20
    learning_rates = [
        0.001, 
        0.0001,
        0.00001
    ]
    epochs_per_save = 5
    batch_sizes = [
        128,
        256,
        512
        ]
    hidden_sizes = [
        32, 
        128, 
        512
    ]
    seed_val = 23
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    train_samples = pickle.load(open(train_pickle_path, "rb"), encoding="latin1")["train"]
    dev_samples = pickle.load(open(dev_pickle_path, "rb"), encoding="latin1")["dev"]

    #batch_sizes.append(len(train_samples))

    # 5 for event types, 3 for BOS, EOS and PADDING, 1 for regression value (delta t)
    if dir_name == "data_hawkes":
        K=5+3+1 
        bos_index = 5
        eos_index = 6
        padding_index = 7

       
    if dir_name == "data_retweet":
        K=3+3+1
        bos_index = 3
        eos_index = 4
        padding_index = 5
    
    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                train_model(train_samples, dev_samples, learning_rate, max_epochs, batch_size, K, hidden_size, bos_index, eos_index, padding_index, dir_name)
    