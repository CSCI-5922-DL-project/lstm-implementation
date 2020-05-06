import os
import pickle
import pandas as pd
runs = [subdir[0] for subdir in os.walk('./saves/data/')][1:]
results = []
for run in runs:
    res = pickle.load(open(run+'/losses.pkl', 'rb'))
    temp = dict()
    temp['train_loss'] = res['results'][-1]['train_loss']
    temp['dev_loss'] = res['results'][-1]['dev_loss']
    temp['event_loss'] = res['results'][-1]['event_loss']
    temp['time_loss'] = res['results'][-1]['time_loss']
    temp['dev_event_loss'] = res['results'][-1]['dev_event_loss']
    temp['dev_time_loss'] = res['results'][-1]['dev_time_loss']

    temp.update(res['args'])

    results.append(temp)

df = pd.DataFrame(results)
df=df[['learning_rate', 'hidden_size', 'batch_size', 'max_epochs', 'train_loss', 'dev_loss', 
'event_loss', 'time_loss', 'dev_event_loss', 'dev_time_loss']]
print(df)