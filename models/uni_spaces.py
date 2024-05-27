import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from models.projector import Reaction_Head_mini
from models.experts import CLAP_GM, IB_audio_ft, InternVL_C
from paths import *


class Uni_Spaces(nn.module):
    def __init__(self):
        super(Uni_Spaces, self).__init__()
        
    def emb_audios(self, audio_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_audios\' method')
    def emb_images(self, image_files:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_images\' method')
    def emb_texts(self, texts:[str])->Tensor:
        raise NotImplementedError('Please define \'emb_texts\' method')
    
class ImageBind(Uni_Spaces):
    def __init__(self):
        super(ImageBind, self).__init__()
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        
    @torch.no_grad()    
    def emb_audios(self, audio_files:[str])->Tensor:
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_files, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.AUDIO])
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(image_files, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.VISION])
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(texts, self.get_device()),
        }
        return F.normalize(self.model(inputs)[ModalityType.TEXT])
    
    def get_device(self):
        return next(self.parameters()).device

class IB_PP(Uni_Spaces):
    def __init__(self, 
                 mode):
        super(IB_PP, self).__init__()
        self.mode = mode
        self.Crat_projectors = self._create_Crat_heads()
        self.uni = self._create_base_space()
        self.clap_experts = CLAP_GM()
        self.factor = 0.5
    
    def _create_Crat_heads(self) -> nn.ModuleList:
        proj_weights = IBPP_PATHS
        projectors = nn.ModuleList([Reaction_Head_mini(in_dim=512, out_dim=1024) for _ in range(7+7)])
        for i in range(len(projectors)):
            projectors[i].load_state_dict(torch.load(proj_weights[i], map_location='cpu'))
            projectors[i].eval()
        return projectors
            
    def _create_base_space(self) -> Uni_Spaces:
        uni_spaces = ImageBind()
        return uni_spaces
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        clap_g_emb, clap_m_emb = self.clap_experts.emb_audios(audio_files)
        uni_audio_emb = self.uni.emb_audios(audio_files)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        uni_audio_emb = F.normalize(uni_audio_emb)
        return self.crat_proj_audio([clap_g_emb, clap_m_emb, uni_audio_emb], factor=self.factor)
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        uni_image_emb = F.normalize(self.uni.emb_images(image_files))
        return uni_image_emb
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        clap_g_emb, clap_m_emb = self.clap_experts.emb_texts(texts)
        uni_text_emb  = self.uni.emb_texts(texts)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        uni_text_emb  = F.normalize(uni_text_emb)
        return self.crat_proj_text([clap_g_emb, clap_m_emb, uni_text_emb], self.factor)
        
    @torch.no_grad()
    def crat_proj_audio(self, x:[torch.Tensor], factor):
        ex_audio = [self.Crat_projectors[i].proj_audio(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_audio = torch.stack(ex_audio, dim=1)
        ex_audio = F.normalize(torch.mean(ex_audio, dim=1), dim=-1)
        return F.normalize(ex_audio*factor + x[-1] * (1-factor), dim=-1)
    
    @torch.no_grad()
    def crat_proj_text(self, x:[torch.Tensor], factor):
        ex_text = [self.Crat_projectors[i].proj_text(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_text = torch.stack(ex_text, dim=1)
        ex_text = F.normalize(torch.mean(ex_text, dim=1), dim=-1)
        return F.normalize(ex_text*factor + x[-1] * (1-factor), dim=-1)

class InternVL_IB(Uni_Spaces):
    def __init__(self):
        super(InternVL_IB, self).__init__()
        self.uni = self._create_base_space()
        self.VL_C_experts = InternVL_C()
        self.Drvt_projectors = self._create_Drvt_heads()
        self.image_factor = 0.2
        self.text_factor  = 0.2
        
    def _create_base_space(self) -> Uni_Spaces:
        return ImageBind()
    
    def _create_Drvt_heads(self) -> nn.ModuleList:
        proj_weights = DRVT_PATHS
        projectors = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=768) for _ in range(7)])
        for i in range(len(projectors)):
            projectors[i].load_state_dict(torch.load(proj_weights[i], map_location='cpu'))
            projectors[i].eval()
        return projectors
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        uni_audio_emb = F.normalize(self.uni.emb_audios(audio_files))
        return uni_audio_emb
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        uni_image_emb = F.normalize(self.uni.emb_images(image_files))
        vl_c_emb      = F.normalize(self.VL_C_experts.emb_images(image_files))
        uni_image_emb = self.drvt_proj_image([uni_image_emb, vl_c_emb], factor=self.image_factor)
        return uni_image_emb
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        uni_text_emb = F.normalize(self.uni.emb_texts(texts))
        vl_c_emb     = F.normalize(self.VL_C_experts.emb_texts(texts))
        uni_text_emb = self.drvt_proj_text([uni_text_emb, vl_c_emb], factor=self.text_factor)
        return uni_text_emb
    
    @torch.no_grad()
    def drvt_proj_image(self, x:[torch.Tensor], factor)->Tensor:
        ex_image = [self.Drvt_projectors[i].proj_image(x[i//7]) for i in range(len(self.Drvt_projectors))]
        ex_image = torch.stack(ex_image, dim=1)
        ex_image = F.normalize(torch.mean(ex_image, dim=1), dim=-1)
        return F.normalize(ex_image*factor + x[-1] * (1-factor), dim=-1)
    
    @torch.no_grad()
    def drvt_proj_text(self, x:[torch.Tensor], factor)->Tensor:
        ex_text = [self.Drvt_projectors[i].proj_text(x[i//7]) for i in range(len(self.Drvt_projectors))]
        ex_text = torch.stack(ex_text, dim=1)
        ex_text = F.normalize(torch.mean(ex_text, dim=1), dim=-1)
        return F.normalize(ex_text*factor + x[-1] * (1-factor), dim=-1)

class InternVL_IB_PP(Uni_Spaces):
    def __init__(self):
        super(InternVL_IB_PP, self).__init__()
        self.Crat_projectors = self._create_Crat_heads()
        self.uni = self._create_base_space()
        self.clap_experts = CLAP_GM()
        self.audio_factor = 0.5
        self.text_factor  = 0.1
    
    def _create_Crat_heads(self) -> nn.ModuleList:
        proj_weights = VLIBPP_PATHS
        projectors = nn.ModuleList([Reaction_Head_mini(in_dim=512, out_dim=768) for _ in range(7+7)])
        for i in range(len(projectors)):
            projectors[i].load_state_dict(torch.load(proj_weights[i], map_location='cpu'))
            projectors[i].eval()
        return projectors
            
    def _create_base_space(self) -> Uni_Spaces:
        uni_spaces = InternVL_IB()
        return uni_spaces
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        uni_audio_emb = self.uni.emb_audios(audio_files)
        clap_g_emb, clap_m_emb = self.clap_experts.emb_audios(audio_files)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        uni_audio_emb = F.normalize(uni_audio_emb)
        return self.crat_proj_audio([clap_g_emb, clap_m_emb, uni_audio_emb], factor=self.audio_factor)
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        uni_image_emb = F.normalize(self.uni.emb_images(image_files))
        return uni_image_emb
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        uni_text_emb = F.normalize(self.uni.emb_texts(texts))
        clap_g_emb, clap_m_emb = self.clap_experts.emb_texts(texts)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        return self.crat_proj_text([clap_g_emb, clap_m_emb, uni_text_emb], factor=self.text_factor)
        
    @torch.no_grad()
    def crat_proj_audio(self, x:[torch.Tensor], factor)->Tensor:
        ex_audio = [self.Crat_projectors[i].proj_audio(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_audio = torch.stack(ex_audio, dim=1)
        ex_audio = F.normalize(torch.mean(ex_audio, dim=1), dim=-1)
        return F.normalize(ex_audio*factor + x[-1] * (1-factor), dim=-1)
    
    @torch.no_grad()
    def crat_proj_text(self, x:[torch.Tensor], factor)->Tensor:
        ex_text = [self.Crat_projectors[i].proj_text(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_text = torch.stack(ex_text, dim=1)
        ex_text = F.normalize(torch.mean(ex_text, dim=1), dim=-1)
        return F.normalize(ex_text*factor + x[-1] * (1-factor), dim=-1)

class InternVL_IB_FT(Uni_Spaces):
    def __init__(self):
        super(InternVL_IB_FT, self).__init__()
        self.uni = self._create_base_space()
        self.VL_C_experts = InternVL_C()
        self.ft_audio_trunk = IB_audio_ft()
        self.Drvt_projectors = self._create_Drvt_heads()
        self.factor = 0.1
        
    def _create_base_space(self) -> Uni_Spaces:
        return ImageBind()
    
    def _create_Drvt_heads(self) -> nn.ModuleList:
        proj_weights = VLIB_PATHS
        projectors = nn.ModuleList([Reaction_Head_mini(in_dim=1024, out_dim=768) for _ in range(7)])
        for i in range(len(projectors)):
            projectors[i].load_state_dict(torch.load(proj_weights[i], map_location='cpu'))
            projectors[i].eval()
        return projectors
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        uni_audio_emb = F.normalize(self.ft_audio_trunk.emb_audios(audio_files))
        uni_audio_emb = self.drvt_proj_audio(uni_audio_emb)
        return uni_audio_emb
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        uni_image_emb = F.normalize(self.uni.emb_images(image_files))
        vl_c_emb      = F.normalize(self.VL_C_experts.emb_images(image_files))
        uni_image_emb = F.normalize(self.drvt_proj_image([uni_image_emb, vl_c_emb], factor=self.factor))
        return uni_image_emb
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        uni_text_emb = F.normalize(self.uni.emb_texts(texts))
        vl_c_emb     = F.normalize(self.VL_C_experts.emb_texts(texts))
        uni_text_emb = F.normalize(self.drvt_proj_text([uni_text_emb, vl_c_emb], factor=self.factor))
        return uni_text_emb
    
    @torch.no_grad()
    def drvt_proj_audio(self, x:torch.Tensor)->Tensor:
        ex_audio = [self.Drvt_projectors[i].proj_audio(x) for i in range(len(self.Drvt_projectors))]
        ex_audio = torch.stack(ex_audio, dim=1)
        ex_audio = F.normalize(torch.mean(ex_audio, dim=1), dim=-1)
        return ex_audio
    
    @torch.no_grad()
    def drvt_proj_image(self, x:[torch.Tensor], factor)->Tensor:
        ex_image = [self.Drvt_projectors[i].proj_image(x[i//7]) for i in range(len(self.Drvt_projectors))]
        ex_image = torch.stack(ex_image, dim=1)
        ex_image = F.normalize(torch.mean(ex_image, dim=1), dim=-1)
        return F.normalize(ex_image*factor + x[-1] * (1-factor), dim=-1)
    
    @torch.no_grad()
    def drvt_proj_text(self, x:[torch.Tensor], factor)->Tensor:
        ex_text = [self.Drvt_projectors[i].proj_text(x[i//7]) for i in range(len(self.Drvt_projectors))]
        ex_text = torch.stack(ex_text, dim=1)
        ex_text = F.normalize(torch.mean(ex_text, dim=1), dim=-1)
        return F.normalize(ex_text*factor + x[-1] * (1-factor), dim=-1)
 
class InternVL_IB_FT_PP(Uni_Spaces):
    def __init__(self, 
                 mode):
        super(InternVL_IB_FT_PP, self).__init__()
        self.mode = mode
        self.Crat_projectors = self._create_Crat_heads()
        self.uni = self._create_base_space()
        self.clap_experts = CLAP_GM()
        self.factor = 0.5
    
    def _create_Crat_heads(self) -> nn.ModuleList:
        proj_weights = VLIBFTPP_PATHS
        projectors = nn.ModuleList([Reaction_Head_mini(in_dim=512, out_dim=768) for _ in range(7+7)])
        for i in range(len(projectors)):
            projectors[i].load_state_dict(torch.load(proj_weights[i], map_location='cpu'))
            projectors[i].eval()
        return projectors
            
    def _create_base_space(self) -> Uni_Spaces:
        uni_spaces = InternVL_IB_FT()
        return uni_spaces
    
    @torch.no_grad()
    def emb_audios(self, audio_files:[str])->Tensor:
        uni_audio_emb = self.uni.emb_audios(audio_files)
        clap_g_emb, clap_m_emb = self.clap_experts.emb_audios(audio_files)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        uni_audio_emb = F.normalize(uni_audio_emb)
        return self.crat_proj_audio([clap_g_emb, clap_m_emb, uni_audio_emb], factor=self.factor)
    
    @torch.no_grad()
    def emb_images(self, image_files:[str])->Tensor:
        uni_image_emb = F.normalize(self.uni.emb_images(image_files))
        return uni_image_emb
    
    @torch.no_grad()
    def emb_texts(self, texts:[str])->Tensor:
        uni_text_emb = F.normalize(self.uni.emb_texts(texts))
        clap_g_emb, clap_m_emb = self.clap_experts.emb_texts(texts)
        clap_g_emb    = F.normalize(clap_g_emb)
        clap_m_emb    = F.normalize(clap_m_emb)
        return self.crat_proj_text([clap_g_emb, clap_m_emb, uni_text_emb], factor=self.factor)
        
    @torch.no_grad()
    def crat_proj_audio(self, x:[torch.Tensor], factor)->Tensor:
        ex_audio = [self.Crat_projectors[i].proj_audio(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_audio = torch.stack(ex_audio, dim=1)
        ex_audio = F.normalize(torch.mean(ex_audio, dim=1), dim=-1)
        return F.normalize(ex_audio*factor + x[-1] * (1-factor), dim=-1)
    
    @torch.no_grad()
    def crat_proj_text(self, x:[torch.Tensor], factor)->Tensor:
        ex_text = [self.Crat_projectors[i].proj_text(x[i//7]) for i in range(len(self.Crat_projectors))]
        ex_text = torch.stack(ex_text, dim=1)
        ex_text = F.normalize(torch.mean(ex_text, dim=1), dim=-1)
        return F.normalize(ex_text*factor + x[-1] * (1-factor), dim=-1)

