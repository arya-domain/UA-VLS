import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
from torchvision import transforms

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode
        self.data = pd.read_excel(csv_path)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])
        
        # Bert
        self.model = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',
                                               output_hidden_states=True,
                                               trust_remote_code=True)

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'img',self.image_list[idx].replace('mask_',''))  # .replace('mask_','')
        image_path = image
        gt = os.path.join(self.root_path,'labelcol', self.image_list[idx])
        gt_path = gt
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token = [token_output['input_ids'].squeeze(0), token_output['attention_mask'].squeeze(0)]
        
        ##########################################################################################
        with torch.no_grad():
            token = self.model(**token_output, output_hidden_states=True,return_dict=True)['hidden_states']
            
            hidden_states = list(token)
            for i in range(len(hidden_states)):
                hidden_states[i] = hidden_states[i].squeeze(0)
            
            token = tuple(hidden_states)
        ##########################################################################################
        
        data = {'image':image, 'gt':gt, 'token': token}
        data = trans(data)
        image, gt, token= data['image'], data['gt'], data['token']
        gt = torch.where(gt==255,1,0)
        
        dtype = gt.dtype
        gt = transforms.ToPILImage()(gt.float())
        gt = gt.convert("L")
        gt = torch.tensor(np.array(gt)).unsqueeze(0)
        gt = torch.where(gt==255,1,0)
        gt = gt.to(dtype)

        return ([image, token], gt) #, (image_path, gt_path, caption)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt"]),

            ])

        return trans


