from models.paths import *
import torch
from models.uni_spaces import Uni_Spaces, ImageBind, IB_PP, InternVL_IB, InternVL_IB_PP, InternVL_IB_FT, InternVL_IB_FT_PP

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu' #
def uni_example(uni: Uni_Spaces):
    audios = ['assets/BBQ.wav', 'assets/toilet.wav', 'assets/train.wav']
    images = ['assets/BBQ.jpeg', 'assets/toilet.jpeg', 'assets/train.jpeg']
    texts  = ['an audio of barbeque', 'an audio of toilet', 'an audio of train']

    ## Adjusting the space combining factors results in spaces with different specialties.
    ## Versatile.
    uni.text_factor=0.1
    uni.audio_factor=0.5
    text_embs  = uni.emb_texts(texts)
    audio_embs = uni.emb_audios(audios)
    image_embs = uni.emb_images(images)

    print("Audio x Image:\n",
        torch.softmax(audio_embs@image_embs.T * 10.0, dim=-1)
    )
    print("Audio x Text:\n",
        torch.softmax(audio_embs@text_embs.T * 10.0, dim=-1)
    )
    print("Image x Text:\n",
        torch.softmax(image_embs@text_embs.T * 10.0, dim=-1)
    )

    ## AT Expertise.
    uni.text_factor=0.5
    uni.audio_factor=0.8

    text_embs  = uni.emb_texts(texts)
    audio_embs = uni.emb_audios(audios)
    image_embs = uni.emb_images(images)

    print("Audio x Image:\n",
        torch.softmax(audio_embs@image_embs.T * 10.0, dim=-1)
    )
    print("Audio x Text:\n",
        torch.softmax(audio_embs@text_embs.T * 10.0, dim=-1)
    )
    print("Image x Text:\n",
        torch.softmax(image_embs@text_embs.T * 10.0, dim=-1)
    )

uni = IB_PP()
uni = uni.to(device)
print('----IBPP----')
uni_example(uni)

uni = InternVL_IB_FT_PP()
uni = uni.to(device)
print('----InternVL_IB_FT_PP----')
uni_example(uni)

# Expected output
# ----IBPP----
# Audio x Image:
#  tensor([[0.7426, 0.1838, 0.0736],
#         [0.0456, 0.9197, 0.0347],
#         [0.0736, 0.0837, 0.8427]], device='cuda:0')
# Audio x Text:
#  tensor([[0.7238, 0.2097, 0.0665],
#         [0.0124, 0.9691, 0.0185],
#         [0.0446, 0.0981, 0.8573]], device='cuda:0')
# Image x Text:
#  tensor([[0.6406, 0.1846, 0.1748],
#         [0.1061, 0.8104, 0.0835],
#         [0.1736, 0.1662, 0.6602]], device='cuda:0')
# Audio x Image:
#  tensor([[0.7371, 0.1669, 0.0960],
#         [0.0357, 0.9237, 0.0406],
#         [0.0641, 0.0967, 0.8392]], device='cuda:0')
# Audio x Text:
#  tensor([[0.6880, 0.2722, 0.0398],
#         [0.0021, 0.9925, 0.0054],
#         [0.0079, 0.0324, 0.9596]], device='cuda:0')
# Image x Text:
#  tensor([[0.6530, 0.2016, 0.1454],
#         [0.0669, 0.8922, 0.0409],
#         [0.1440, 0.1134, 0.7426]], device='cuda:0')

# ----InternVL_IB_FT_PP----
# Audio x Image:
#  tensor([[0.6601, 0.2232, 0.1167],
#         [0.0568, 0.8933, 0.0499],
#         [0.0873, 0.1187, 0.7941]], device='cuda:0')
# Audio x Text:
#  tensor([[0.7360, 0.1836, 0.0804],
#         [0.1283, 0.7124, 0.1593],
#         [0.1276, 0.1832, 0.6893]], device='cuda:0')
# Image x Text:
#  tensor([[0.5094, 0.2608, 0.2298],
#         [0.1742, 0.6009, 0.2249],
#         [0.2390, 0.2895, 0.4715]], device='cuda:0')
# Audio x Image:
#  tensor([[0.6730, 0.2183, 0.1087],
#         [0.0376, 0.9099, 0.0525],
#         [0.0864, 0.2038, 0.7098]], device='cuda:0')
# Audio x Text:
#  tensor([[0.6963, 0.2787, 0.0250],
#         [0.0101, 0.9784, 0.0115],
#         [0.0324, 0.0571, 0.9105]], device='cuda:0')
# Image x Text:
#  tensor([[0.5324, 0.2517, 0.2159],
#         [0.0732, 0.8440, 0.0828],
#         [0.1844, 0.2028, 0.6128]], device='cuda:0')