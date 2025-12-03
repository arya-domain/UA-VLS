import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import Decoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel


class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        # embeds = output['pooler_output'].squeeze()
        # project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":output}


class UA_VLS(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(UA_VLS, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = Decoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = Decoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = Decoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text_embeds = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, _ = image_output['feature'], image_output['project']

        if len(image_features[0].shape) == 4: 
            if len(image_features) == 5:  # convnext
                image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 
    
        
        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
