import glob 
import os 
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from typing import List, Union, Tuple, Any
import torch.nn as nn
from torch.utils.data import DataLoader


def remove_previous_files(path):
    existing_files = glob.glob(f'{path}*')
    for f in existing_files:
        os.remove(f)
    return

def validate_model(physio_model:nn.Module, 
                   val_dataloader:DataLoader, 
                   criterion, 
                   device):
    """ Calculate the accuarcy, loss and f1 score of the model on the given dataset
    (no prints, no save)"""
    # Validation phase
    physio_model.eval() 

    #Metrics to evaluate
    val_correct = 0
    val_total = 0
    val_physio_loss = 0.0

    tp = 0
    fn = 0
    fp = 0

    with torch.no_grad():
        for val_data in val_dataloader:
            #Load inputs
            val_inputs, val_labels = val_data
            val_inputs = val_inputs.reshape(val_inputs.shape[0],1,val_inputs.shape[1])
            
            val_inputs = val_inputs.to(device, dtype=torch.float)
            val_labels = val_labels.to(device)
        
            #Apply model
            val_physio_outputs = physio_model(val_inputs)

            #Update metrics
            val_physio_loss += criterion(val_physio_outputs, val_labels)
            _,val_predicted = torch.max(val_physio_outputs.data, 1)

            batch_size = len(val_labels)
            for ind in range(batch_size):
                pred = val_predicted[ind].item()
                true = val_labels[ind].item()


                if (pred == 1):
                    if (pred == true):
                        tp += 1
                    else:
                        fp += 1
                if (pred == 0):
                    if (true == 1):
                        fn += 1

            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    #Final results
    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = ((val_physio_loss)) / len(val_dataloader)
    f1_score2 = (2 * tp) / (2 * tp + fn + fp)

    return val_accuracy, avg_val_loss, f1_score2