# Unleashing Degradation-Carrying Features in Symmetric U-Net: Simpler and Stronger Baselines for All-in-One Image Restoration
Wenlong Jiao, Heyang Lee, Ping Wang, Pengfei Zhu, Qinghua Hu, Dongwei Ren
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2512.10581" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2512.10581-red' alt='arxiv'>
  </a>
</div>

***
>**Abstract**: All-in-one image restoration aims to handle diverse degradations (e.g., noise, blur, adverse weather) within a unified framework, yet existing methods increasingly rely on complex architectures (e.g., Mixture-of-Experts, diffusion models) and elaborate degradation prompt strategies. In this work, we reveal a critical insight: well-crafted feature extraction inherently encodes degradation-carrying information, and a symmetric U-Net architecture is sufficient to unleash these cues effectively. By aligning feature scales across encoder-decoder and enabling streamlined cross-scale propagation, our symmetric design preserves intrinsic degradation signals robustly, rendering simple additive fusion in skip connections sufficient for state-of-the-art performance. Our primary baseline, SymUNet, is built on this symmetric U-Net and achieves better results across benchmark datasets than existing approaches while reducing computational cost. We further propose a semantic enhanced variant, SE-SymUNet, which integrates direct semantic injection from frozen CLIP features via simple cross-attention to explicitly amplify degradation priors. Extensive experiments on several benchmarks validate the superiority of our methods. Both baselines SymUNet and SE-SymUNet establish simpler and stronger foundations for future advancements in all-in-one image restoration.
***

## QuickRun

### Install

```bash
conda create -n symunet python=3.12
conda activate symunet
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
pip install --no-build-isolation -e .
```
### Training

```bash
CUDA_VISIBLE_DEVICES=x,x torchrun --nproc_per_node=x --master_port=xxxxx basicsr/train.py -opt options/train/AllInOne/xxx.yml --launcher pytorch
```

### Evaluation
```bash
CUDA_VISIBLE_DEVICES=x,x torchrun --nproc_per_node=x --master_port=xxxxx basicsr/test.py -opt options/test/AllInOne/xxx.yml --launcher pytorch
```
