import torch
from torch import nn, Tensor
from torch.nn import functional as F
import laion_clap
from projector import Reaction_Head_mini
from paths import *
from imagebind_audio import imagebind_huge_audio

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import os

class CLAP_GM(nn.modules):
    def __init__(self):
        super(CLAP_GM, self).__init__()
        self.clap_g = laion_clap.CLAP_Module(enable_fusion=True, device='cpu')
        self.clap_g.load_ckpt(CLAP_G_PATH)
        self.clap_m = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device='cpu')
        self.clap_m.load_ckpt(CLAP_M_PATH)
        
    def emb_audios(self, audio_files:[str]):
        clap_g_emb = F.normalize(self.clap_g.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True))
        clap_m_emb = F.normalize(self.clap_m.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True))
        return clap_g_emb, clap_m_emb
    
    def emb_texts(self, texts:[str]):
        clap_g_emb = F.normalize(self.clap_g.get_text_embedding(x=texts, use_tensor=True))
        clap_m_emb = F.normalize(self.clap_m.get_text_embedding(x=texts, use_tensor=True))
        return clap_g_emb, clap_m_emb

class BOOST_AUDIO_IB(torch.nn.Module):
    def __init__(self):
        super(BOOST_AUDIO_IB, self).__init__()
        self.projectors = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=768) for _ in range(7)])

    def forward(self, x:torch.Tensor) -> Tensor:
        # with torch.no_grad():
        ex_audio = [self.projectors[i].proj_audio(x) for i in range(len(self.projectors))]
        ex_audio = torch.stack(ex_audio, dim=1)
        return F.normalize(torch.mean(ex_audio, dim=1), dim=-1)
    
    def get_device(self):
        return next(self.parameters()).device

class ib_with_uni_head(torch.nn.Module):
    def __init__(self, ib):
        super(ib_with_uni_head, self).__init__()
        self.ib     = ib
        self.ib_head  = BOOST_AUDIO_IB()

    def get_device(self):
        return next(self.parameters()).device

class IB_audio_ft(nn.modules):
    def __init__(self):
        super(IB_audio_ft, self).__init__()
        self.model = ib_with_uni_head(imagebind_huge_audio(pretrained=False))
        state_dict = torch.load(IB_FT_PATH, map_location='cpu')
        self.model.load_state_dict(state_dict)
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str]):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_files, self.get_device())
        }
        embeddings = self.model.ib(inputs)[ModalityType.AUDIO]
        audio_embs.append(embeddings)

        audio_embs = torch.cat(audio_embs, dim=0)
        audio_embs = F.normalize(audio_embs, dim=-1)
        audio_embs = self.model.ib_head(audio_embs)
    
    def get_device(self):
        return next(self.parameters()).device
    
class InternVL_C(nn.modules):
    def __init__(self):
        super(InternVL_C, self).__init__()
        self.model = AutoModel.from_pretrained(
                    VLC_PATH,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True)
        self.processor = CLIPImageProcessor.from_pretrained(VLC_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(
            VLC_PATH, use_fast=False, add_eos_token=True,local_files_only=True)
        
    def emb_images(self, image_files:[str]):
        imgs = []
        for p in image_files:
            if os.path.exists(p):
                imgs.append(Image.open(p).convert('RGB'))
        
        image_data = self.processor(images=image_data, return_tensors='pt').pixel_values.to(torch.bfloat16).to(self.get_device())
        image_features = self.model.encode_image(image_data, mode='InternVL-C')
        return F.normalize(image_features)
        
    def emb_texts(self, texts:[str]):
        texts = ['summarize:' + t for t in texts]
        texts = self.tokenizer(texts, return_tensors='pt', max_length=80,
                truncation=True, padding='max_length').input_ids.to(self.get_device())

        text_embs = self.model.encode_text(texts)
        return F.normalize(text_embs)
    
    
    def get_device(self):
        return next(self.model.parameters()).device
    
