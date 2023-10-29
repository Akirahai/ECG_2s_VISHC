
import torch
from tqdm import tqdm
import numpy as np
from utils import static
from sklearn.metrics import *

def test(dataloader, net, criterion, device):
    print('Testing...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
        
    # print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = static.cal_f1s(y_trues, y_scores)
    
    avg_f1 = np.mean(f1s)
    
    
    return f1s, avg_f1, y_trues, y_scores
    
    