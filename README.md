<h2 align="center"> <a href="https://arxiv.org/pdf/2405.04883">【ICML 2024 🔥】FreeBind: Free Lunch in Unified Multimodal Space via Knowledge Fusion </a> </h2>

[**Zehan Wang**](https://zehanwang01.github.io/) · [**Ziang Zhang**]() · [**Xize Cheng**](https://exgc.github.io/) · [**Rongjie Huang**](https://rongjiehuang.github.io/) · [**Luping Liu**](https://luping-liu.github.io/) · [**Zhenhui Ye**]() · [**Haifeng Huang**]() · [**Yang Zhao**]() · [**Tao Jin**]() · [**Peng Gao**]() · [**Zhou Zhao**]()

FreeBind is an efficient unified multimodal space enhancement strategy. Built upon ImageBind, we build an audio-visual-text representation that outperforms ImageBind by a large margin.

## News
- `2024/05/02`: FreeBind is accepted by ICML2024.

## To-Do List
- [ ] Inference code
- [ ] Model zoo (audio-image-text)

Before 31th, May. Please stay tuned.

## Highlight

### Fast and efficient pre-trained unified space enhancement
Intergrating CLIPs, CLAPs to ImageBind brings comprehensively improved audio-image-text space.

### Flexible post-training customization
Adjusting the space combining factors results in spaces with different specialties. 

We provide two default settings, **_AT Expertise._** (Better audio-text version, surpass advanced CLAPs), and **_Versatile._** (Balanced version, state-of-the-art audio-image and image-text performance)


## Performance

Comparison on zero-shot cross-modal retrieval tasks: 

<p align="left">
<img src="assets/figure2.png" width=100%>
</p>

Comparison on zero-shot classification tasks:

<p align="left">
<img src="assets/figure1.png" width=50%>
</p>

## Usage
We are working hard to prepare our code and weights, which will be released in **May.**

## Citation
If you find this project useful, please consider giving a star and citation:

```bibtex
@misc{wang2023extending,
      title={Extending Multi-modal Contrastive Representations}, 
      author={Zehan Wang and Ziang Zhang and Luping Liu and Yang Zhao and Haifeng Huang and Tao Jin and Zhou Zhao},
      year={2023},
      eprint={2310.08884},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{wang2023connecting,
      title={Connecting Multi-modal Contrastive Representations}, 
      author={Zehan Wang and Yang Zhao and Xize Cheng and Haifeng Huang and Jiageng Liu and Li Tang and Linjun Li and Yongqi Wang and Aoxiong Yin and Ziang Zhang and Zhou Zhao},
      year={2023},
      eprint={2305.14381},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!--
**zehanwang01/FreeBind** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
