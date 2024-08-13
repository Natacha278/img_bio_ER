import torch
import torch.nn as nn

class BioImgFusNet(nn.Module):
    def __init__(self, bio_model, img_model, fus_model):
        self.bio_model = bio_model
        self.img_model = img_model
        self.fus_model = fus_model

    def forward(self, source, target):
        img_domain1, label_domain1, signal_domain1 = source
        img_domain2, label_domain2, signal_domain2 = target

        img_pred_domain1, transfer_loss, domain1_feature = self.img_model(img_domain1, img_domain2)

        bio_output_domain1 = self.bio_model.predict(signal_domain1)
        fus_input_domain1 = torch.cat((img_pred_domain1,bio_output_domain1), dim = 1)
        fus_output_domain1  = self.fus_model(fus_input_domain1)

        return fus_output_domain1, transfer_loss, domain1_feature 
    


    def predict(self, x):
        img, label, signal = x

        img_output = self.img_model.predict(img)
        bio_output = self.bio_model.predict(signal)
        fus_input = torch.cat((img_output,bio_output), dim = 1)
        fus_output = self.fus_model(fus_input)

        return fus_output
