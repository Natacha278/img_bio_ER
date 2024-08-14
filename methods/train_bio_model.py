import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import List, Union, Tuple, Any
import statistics
from sklearn.model_selection import train_test_split
from utils.util import remove_previous_files, validate_model


def train_bio_model(bio_model:nn.Module,
                    bio_optimizer,
                    train_loader:DataLoader,
                    val_loader:DataLoader,
                    num_epochs:int = 100,
                    weight_path:str = None,
                    scheduler = None):
    """ Train the given torch Model: 
       - Basic GD on the train data, val data for model selection
       - Returns trained model, best_val_acc, best_val_f1, best_train_acc, best_epoch
       - eventually save results and weights if path given"""
    
    check_every = 1
    best_val_acc = 0
    best_val_f1 = 0
    best_train_acc = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()


    best_model = copy.deepcopy(bio_model)

    #Train the model
    for epoch in range(num_epochs):
        bio_model.train()
        
        #Accuracy and epoch loss
        running_loss = 0
        correct = 0
        total = 0
        
        for i,(physio_batch, labels) in enumerate(train_loader):
            #Train the model
            bio_optimizer.zero_grad()

            physio_batch = physio_batch.reshape(physio_batch.shape[0],1,physio_batch.shape[1])
            physio_batch = physio_batch.to(device, dtype=torch.float)
            labels = labels.to(device)
            
            physio_outputs = bio_model(physio_batch)
            
            physio_loss = criterion(physio_outputs, labels)
            
            physio_loss.backward()
            bio_optimizer.step()

            #Evaluate
            running_loss += physio_loss.item()
            
            _, physio_predicted = torch.max(physio_outputs.data, 1)
            total += labels.size(0)

            correct += (physio_predicted == labels).sum().item()

        if scheduler:
            scheduler.step()

        #Save best model (using the validation set)
        if epoch % check_every == 0:
                val_acc, val_loss, val_f1 = validate_model(bio_model, val_loader, criterion, device)
                

                if val_acc >= best_val_acc:
                    
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_train_acc = 100 * correct / total
                    best_epoch = epoch+1
                    best_model = copy.deepcopy(bio_model)

                    if weight_path:
                        remove_previous_files(weight_path)
                        model_save_path = f'{weight_path}{round(best_val_acc,2)}.pth'
                        torch.save(bio_model.state_dict(), model_save_path)


                    

    print("Finished Training")

    train_accuracy = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    print(f'Training accuracy: {train_accuracy}%')
    print(f'Training loss: {avg_train_loss}')

    print("Best model saved at epoch: ", best_epoch)
    print("Best validation accuracy: ", best_val_acc)

    
    return best_model, best_val_acc, best_val_f1, best_train_acc, best_epoch
