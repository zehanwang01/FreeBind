import os
current_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(os.path.abspath(os.path.join(current_dir, os.pardir)), 'checkpoints')
print('ckpt_path:', ckpt_path)

VLIBFTPP_PATHS = []
VLIBPP_PATHS = []
IBPP_PATHS = []
VLIB_PATHS = []
CLAP_G_PATH = os.path.join(ckpt_path, 'laion_clap_fullset_fusion.pt')
CLAP_M_PATH = os.path.join(ckpt_path, 'music_speech_audioset_epoch_15_esc_89.98.pt')
IB_FT_PATH = os.path.join(ckpt_path, 'InternVL_IB_audio_with_head.pt')
VLC_PATH = os.path.join(ckpt_path, 'InternVL-14B-224px')

data_modes = ['mix', 'A', 'T', 'V', 'AT', 'AV', 'TV']

for dm in data_modes:
    IBPP_PATHS.append(os.path.join(ckpt_path, f'IB_PP_G_Proj/{dm}/best.pt'))
for dm in data_modes:
    IBPP_PATHS.append(os.path.join(ckpt_path, f'IB_PP_M_Proj/{dm}/best.pt'))

for dm in data_modes:
    VLIB_PATHS.append(os.path.join(ckpt_path, f'InternVL_IB_Proj/{dm}/best.pt'))

for dm in data_modes:
    VLIBPP_PATHS.append(os.path.join(ckpt_path, f'InternVL_IB_PP_G_Proj/{dm}/best.pt'))
for dm in data_modes:
    VLIBPP_PATHS.append(os.path.join(ckpt_path, f'InternVL_IB_PP_M_Proj/{dm}/best.pt'))

for dm in data_modes:
    VLIBFTPP_PATHS.append(os.path.join(ckpt_path, f'InternVL_IB_FTPP_G_Proj/{dm}/best.pt'))
for dm in data_modes:
    VLIBFTPP_PATHS.append(os.path.join(ckpt_path, f'InternVL_IB_FTPP_M_Proj/{dm}/best.pt'))
