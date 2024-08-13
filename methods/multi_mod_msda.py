'''Multi_Subject in Domain Adaptation.

'''


import argparse

import os, sys
import numpy as np
import torch
from torchmetrics import Accuracy
from torch import nn

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from networks.fus_net import BioImgFusNet
from models.bio_fus_models import Conv1D_model, Conv_LSTM, Fus_model
from networks.transfer_net import TransferNet
from utils.common import *

from datasets.base_dataset import BaseDataset
import torch.nn.functional as F

# from models.resnet_fer_modified import *
from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms
from scipy.special import softmax

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bio_img(data):
    img, label, signal = data
    img, label = img.cuda(), label.cuda()
    if len(signal.shape) < 3:
        signal = signal.reshape(signal.shape[0],1,signal.shape[1])
    signal = signal.to(device, dtype=torch.float)
    return img, label, signal
    

def main(args, target_subject):

    '''
    BIOVID DATSET:
        - Treat each subject as a domain
        - NUM_SUBJECTS indicates number of sources + single target subject
        - TARGET_SUBJECT indicate which subject needed to be treated as a target subject 
        - create src and traget file name
    '''
    source_list_name = load_biovid_src_subs(topk=args.src_sub_topk)

    subject_list = source_list_name
    subject_list.append(target_subject)
    print(subject_list)

    subject_list = write_srcs_tar_txt_files_using_list(args.sub_domains_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, args.oracle_setting)
    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.sub_domains_datasets_path, target_subject, args.n_class, args.oracle_setting)

    src_subs_loader, src_subs_loader_val_loader, _ = BaseDataset.load_bio_img_dataset(args.pain_db_root_path, srcs_file_name, None, args.batch_size, phase='src')
    tar_sub_loader, tar_sub_val_loader, tar_sub_test_loader  = BaseDataset.load_bio_img_dataset(args.pain_db_root_path, tar_file_name, None, args.batch_size, phase='tar')

    dataloaders = {
        'tar': tar_sub_loader,
        'tar_val': tar_sub_val_loader,
        'tar_test': tar_sub_test_loader,
        'src_subs': src_subs_loader,
        'src_subs_val': src_subs_loader_val_loader,
    }

    ''' 
        FOR BIOVID
    '''
    source_model_name = srcs_file_name.split('.')[0]
    tar_model_name = tar_file_name.split('.')[0] 

    #Load img model
    lr_rate = 0.0001
    transfer_model, optimizer = initialize_model(args, source_model_name, args.train_src_subs, lr_rate, args.transfer_loss, args.n_class)

    #Load bio model
    bio_model = Conv1D_model(num_classes= args.n_class)
    bio_model_path = config.MODEL_BIO_PATH
    bio_model.load_state_dict(torch.load(bio_model_path))
    bio_model.to(device)
    for parameter in bio_model.parameters():
        parameter.requires_grad = False

    #Load fusion model
    fus_model_path = config.MODEL_FUS_PATH
    trained_fus_model = Fus_model(num_classes= args.n_class)
    trained_fus_model.to(device)
    trained_fus_model.load_state_dict(torch.load(fus_model_path))
    for parameter in trained_fus_model.parameters():
        parameter.requires_grad = False

    multi_mod_model = BioImgFusNet(bio_model,transfer_model,trained_fus_model)
    
    #TRAINING
    if not args.target_evaluation_only:        
        train(dataloaders, multi_mod_model, optimizer, source_model_name, tar_model_name, args)
    
    #TESTING
    target_trained_model = torch.load(config.CURRENT_DIR + '/' + tar_model_name  + '_load.pt')
    transfer_model.load_state_dict(target_trained_model['model_state_dict'])
    models_DA = (transfer_model, bio_model, trained_fus_model)

    acc_test = test_fus(models_DA, dataloaders['tar_test'], args.batch_size, args.n_class, args.is_pain_dataset)
    print(f'Target Test fusion Accuracy: {acc_test}')



def initialize_model(args, source_model_name, train_src_subs, lr_rate, transfer_loss, n_class):
    transfer_model = TransferNet(n_class, transfer_loss=transfer_loss, base_net=args.back_bone).cuda()

    optimizer = torch.optim.SGD([
        {'params': transfer_model.base_network.parameters()},
        {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * lr_rate},
        {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * lr_rate},
    ], lr=lr_rate, momentum=0.9, weight_decay=5e-4)

    # -- load trained source model and train only target
    if not train_src_subs:
        source_trained_model = torch.load(source_model_name + '_load.pt')
        transfer_model.load_state_dict(source_trained_model['model_state_dict'])
        optimizer.load_state_dict(source_trained_model['optimizer_state_dict'])
    
    return transfer_model, optimizer

def train(dataloaders, model, optimizer, source_model_name, tar_model_name, args):
    target_loader, src_subs_loader  = dataloaders['tar'], dataloaders['src_subs'] 
    target_val_loader, src_subs_val_loader = dataloaders['tar_val'],dataloaders['src_subs_val']
    len_target_loader = len(target_loader)
    len_src_subs = len(src_subs_loader)
    
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------ ------------------------------------- #
    # Training of labeled multi-subjects source domains 

    # ------------------------------------------ ------------------------------------- #
    if args.train_src_subs:
        n_batch = len_src_subs
        print('\n ------ Start Training of Source Domain ------ \n')
        model = train_multi_model(args.source_epochs, model, src_subs_loader, src_subs_loader, optimizer, criterion, n_batch, source_model_name, src_subs_val_loader, train_source=True)
    
    # ------------------------------------------ ------------------------------------- #
    # Adaptation to the target subject
    #
    # ------------------------------------------ ------------------------------------- #
    if  args.train_tar_sub:
        print('\n ------ Start Training of Target Domain ------ \n')
        n_batch = min(len_src_subs, len_target_loader)
        train_multi_model(args.target_epochs, model, src_subs_loader, target_loader, optimizer, criterion, n_batch, tar_model_name, target_val_loader, train_source=False)

def train_multi_model(n_epoch, multi_mod_model, data_loader1, data_loader2, optimizer, criterion, n_batch, trained_model_name, val_loader, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    threshold = 0.90

    
    tar_loader_for_PL = data_loader2
    # train_source = False # remove this line ; ITS ONLY THERE TO PERFORM EXPERIMENTS ON THE MMD LOSS FOR DOMAIN SHIFT FOR GDA FOR DGA-1033 PRESENTATION
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0
        calculate_tar_pl_ce = True

        if train_source is False:
            if e % 20 == 0:
                _data_arr, _prob_arr, _label_arr, _gt_arr, _signal_arr = generate_tar_aug_conf_pl(multi_mod_model, tar_loader_for_PL, threshold)
                
                for i in range(0,1):
                    target_wth_gt_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr), torch.tensor(_signal_arr))
                    tar_loader, _ = BaseDataset.load_tar_acpl_data(target_wth_gt_labels, args.batch_size, split=False)
                    _data_arr, _prob_arr, _label_arr, _gt_arr, _signal_arr = generate_tar_aug_conf_pl(multi_mod_model, tar_loader, threshold)
                    
                target_wth_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_label_arr), torch.tensor(_signal_arr),torch.tensor(_prob_arr), torch.tensor(_gt_arr))
                tar_loader, _ = BaseDataset.load_tar_acpl_data(target_wth_labels, args.batch_size, split=False)
                calculate_tar_pl_ce = True
                data_loader2 = tar_loader

                target_for_PL = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr), torch.tensor(_signal_arr))
                tar_loader_for_PL, _ = BaseDataset.load_tar_acpl_data(target_for_PL, args.batch_size, split=False)
                threshold = threshold - 0.02

        #############################################
        multi_mod_model.train()

        count = 0
        total_mmd = 0
        srcs_avg_features = []
        for (domain1, domain2) in zip(data_loader1, data_loader2):

            count = count + 1 

            img_domain1, label_domain1, signal_domain1 = get_bio_img(domain1)

             # defining for domain-2
            if train_source:
                img_domain2, label_domain2, signal_domain2 = get_bio_img(domain2)
            else:
                img_domain2, label_domain2, signal_domain2, _,_ = domain2
                img_domain2, label_domain2, signal_domain2  = get_bio_img((img_domain2, label_domain2, signal_domain2))

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            img_domain1 = img_domain1.float()
            img_domain2 = img_domain2.float()

            label_source_pred, transfer_loss, domain1_feature = multi_mod_model((img_domain1, label_domain1, signal_domain1),(img_domain2, label_domain2, signal_domain2))
            clf_loss = criterion(label_source_pred, label_domain1)
            transfer_loss = transfer_loss.detach().item() if transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + transfer_loss

            total_mmd += transfer_loss 

            # adding target loss with source loss
            if train_source:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = multi_mod_model((img_domain2, label_domain2, signal_domain2),(img_domain1, label_domain1, signal_domain1))
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) +  transfer_loss_domain2

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce:
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = multi_mod_model((img_domain2, label_domain2, signal_domain2),(None, None, None))
                    clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                    combine_loss = clf_loss_domain2 + loss
                    optimizer.zero_grad()
                    combine_loss.backward()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() + train_loss_clf
            train_loss_transfer = transfer_loss + train_loss_transfer
            train_loss_total = combine_loss.detach().item() + train_loss_total

            # target loss_clf
            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

        acc = test_fus(multi_mod_model, val_loader, args.batch_size, args.n_class, args.is_pain_dataset)
        
        writelogs = f'Epoch: [{e:2d}/{n_epoch}], train_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}' 
        print(writelogs)
        if train_source:
            writelogs_src = f'Epoch: [{e:2d}/{n_epoch}], train_loss_clf_domain2: {train_loss_clf_domain2/n_batch:.4f}, train_loss_transfer: {train_loss_transfer_domain2/n_batch:.4f}, train_loss_total_domain2: {train_loss_total_domain2/n_batch:.4f}, acc: {acc:.4f}' 
            print(writelogs_src)

        if best_acc < acc:
            best_acc = acc

            torch.save(multi_mod_model.img_model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + '.pkl')
            # torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + 'FER_model.pt')
            torch.save({'model_state_dict': multi_mod_model.img_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config.CURRENT_DIR + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, config.CURRENT_DIR + '/' + trained_model_name + '_features.pt')
            stop = 0
    print("total mmd: ",total_mmd)
    # visualize_tsne(clusters_by_label)
    print("Best Acc: ", best_acc)
    return multi_mod_model

##################################################################################################################
##################################################################################################################
def test_fus(multi_mod_model, target_test_loader, batch_size, n_class, is_pain_dataset=False):
    multi_mod_model.eval()

    correct = 0
    corr_acc_top1 = 0
    len_target_dataset = len(target_test_loader) if is_pain_dataset else len(target_test_loader.dataset) 

    with torch.no_grad():
        for datas in target_test_loader:
            img, label, signal = get_bio_img(datas)
            s_output = multi_mod_model((img,label,signal))

            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == label)

            # calculate Top 1 Percent accuracy
            # acc_top1 = Accuracy(top_k=1).to(device)
            acc_top1 = Accuracy(task='multiclass', num_classes=n_class, top_k=1).to(device)
            corr_acc_top1 += acc_top1(s_output, label)

    batch_samples = len_target_dataset if len_target_dataset == len(target_test_loader) else len_target_dataset / int(batch_size)
    acc_top1 = corr_acc_top1/batch_samples

    return acc_top1

def test_img(model, target_test_loader, batch_size, n_class, is_pain_dataset=False):
    model.eval()
    correct = 0
    corr_acc_top1 = 0
    len_target_dataset = len(target_test_loader) if is_pain_dataset else len(target_test_loader.dataset) 

    with torch.no_grad():
        for datas in target_test_loader:
            img, label, signal = get_bio_img(datas)
            s_output = model.predict(img.float())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == label)

            # calculate Top 1 Percent accuracy
            # acc_top1 = Accuracy(top_k=1).to(device)
            acc_top1 = Accuracy(task='multiclass', num_classes=n_class, top_k=1).to(device)
            corr_acc_top1 += acc_top1(s_output, label)

    batch_samples = len_target_dataset if len_target_dataset == len(target_test_loader) else len_target_dataset / int(batch_size)
    acc_top1 = corr_acc_top1/batch_samples

    return acc_top1

# ------------------ ------------------------------------------------------------------ #
# Creating target Tpl and Tcl dictionaries, containing data + labels + class_prob 
# ------------------ ------------------------------------------------------------------ #
class CustomHorizontalFlip:
    def __call__(self, image):
        return torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ]) (image)

def generate_tar_aug_conf_pl(multi_mod_model, target_domain, threshold):
    multi_mod_model.eval()
    print("\n**** Generating Augmented Confident Target Pseudo-labels **** \n")
    
    data_arr = []
    prob_arr = []
    label_arr = []
    gt_arr = []
    # correct_pred_arr = []

    conf_data_arr = []
    conf_pred_arr = []
    conf_label_arr = []
    conf_signal_arr = []

    horizontal_flip_transform = CustomHorizontalFlip()
    with torch.no_grad():
        for datas in target_domain:
            img, label, signal = get_bio_img(datas)
            s_output = multi_mod_model((img,label,signal))
            pred = torch.max(s_output, 1)[1]

            augmented_images = [horizontal_flip_transform(tensor) for tensor in img]

            augmented_batch_tensor = torch.stack(augmented_images)
            augmented_batch_tensor=augmented_batch_tensor.cuda()
            s_output_aug = multi_mod_model((augmented_batch_tensor,label,signal))
            pred_aug = torch.max(s_output_aug, 1)[1]

            softmax_pred_aug = get_target_pred_val(s_output_aug.detach().cpu().numpy(), pred_aug.detach().cpu().numpy())
            softmax_pred = get_target_pred_val(s_output.detach().cpu().numpy(), pred.detach().cpu().numpy())
            soft_pred_avg = np.add(softmax_pred, softmax_pred_aug)/2
            
            np_prob = np.array(soft_pred_avg)

            conf_indxs = np.where(np_prob > threshold)[0]
            if len(conf_indxs) > 0:
                conf_prob = np_prob[conf_indxs]
                conf_pred_arr.extend(conf_prob)

                np_data = np.array(img.detach().cpu().numpy())
                conf_data = np_data[conf_indxs]
                conf_data_arr.extend(conf_data)

                #Save biosignal data
                np_signal = np.array(signal.detach().cpu().numpy())
                conf_signal = np_signal[conf_indxs]
                conf_signal_arr.extend(conf_signal)

                # take labels from the prediction that has the highest softmax prob 
                soft_np = np.array(softmax_pred)
                soft_aug_np = np.array(softmax_pred_aug)
                
                np_label = np.array(pred.detach().cpu().numpy())
                np_gt_label = np.array(label.detach().cpu().numpy())
                aug_np_label = np.array(pred_aug.detach().cpu().numpy())
                conf_label = []
                for conf_indx in conf_indxs:
                    if soft_np[conf_indx] > soft_aug_np[conf_indx] :
                        label = np_label[conf_indx] 
                    else :
                        label = aug_np_label[conf_indx]

                    conf_label.append(label)
                    conf_label_arr.append(label)
                    
                gt_arr.extend(np_gt_label[conf_indxs])
                conf_label = np.array(conf_label)
                
            # creating Tpl; which contains all the data 
            data_arr.extend(img.detach().cpu().numpy())
            prob_arr.extend(softmax_pred_aug)
            label_arr.extend(pred.detach().cpu().numpy())

    """
    The np.asarray() function in NumPy is used to convert an input array-like object (such as a list, 
    tuple, or ndarray) into an ndarray. It creates a new ndarray if the input is not already an ndarray, 
    and it returns the input as is if it is already an ndarray.
    """
    conf_data_arr = np.asarray(conf_data_arr)
    conf_pred_arr = np.asarray(conf_pred_arr)
    conf_label_arr = np.asarray(conf_label_arr)
    conf_signal_arr = np.asarray(conf_signal_arr)

    data_arr = np.asarray(data_arr)
    prob_arr = np.asarray(prob_arr)
    label_arr = np.asarray(label_arr)
    gt_arr = np.asarray(gt_arr)

    return conf_data_arr, conf_pred_arr, conf_label_arr, gt_arr, conf_signal_arr

def get_target_pred_val(s_output, target):
    tar_values = []
    softmax_pred = softmax(s_output, axis=1)
    for i in range(0, len(target)):
        tar_values.append(softmax_pred[i][target[i]])
    return tar_values

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on FER')
    arg_parser.add_argument('--sub_domains_datasets_path', type=str, default=config.BIOVID_SUBS_PATH)
    arg_parser.add_argument('--sub_domains_label_path', type=str, default=config.BIOVID_REDUCE_LABEL_PATH)
    arg_parser.add_argument('--pain_db_root_path', type=str, default=config.BIOVID_PATH)

    arg_parser.add_argument('--transfer_loss', type=str, default='mmd')
    arg_parser.add_argument('--n_class', type=int, default=5)
    arg_parser.add_argument('--batch_size', type=int, default=16) 
    arg_parser.add_argument('--source_epochs', type=int, default=10)
    arg_parser.add_argument('--target_epochs', type=int, default=50)
    arg_parser.add_argument('--back_bone', default="resnet18", type=str)
    arg_parser.add_argument('--oracle_setting', type=str, default=False)

    arg_parser.add_argument('--train_src_subs', type=str, default=False)
    arg_parser.add_argument('--train_tar_sub', type=str, default=True)
    arg_parser.add_argument('--target_evaluation_only', type=bool, default=False)
    arg_parser.add_argument('--src_sub_topk', type=str, default=None)
    arg_parser.add_argument('--is_pain_dataset', type=bool, default=True)
    args = arg_parser.parse_args()

    '''
        List of target subjects
    '''

    target_list_name = ["081014_w_27", "101609_m_36", "112009_w_43", "091809_w_43","071309_w_21",
                "073114_m_25", "080314_w_25", "073109_w_28", "100909_w_65", "081609_w_40"]


    for tar_sub in target_list_name:
        main(args, tar_sub)


    
