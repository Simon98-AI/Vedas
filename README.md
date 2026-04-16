<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Visual Enhanced Depth Scaling for Multimodal Latent Reasoning
</div>

<div align="center">

<!-- Paper Link -->

<a href="">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>

<!-- HuggingFace Models -->

<a href="">
    <img src="https://img.shields.io/badge/HuggingFace-Models-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Models">
  </a>

<a href="">
    <img src="https://img.shields.io/badge/HuggingFace-Papers-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Papers">
  </a>

</div>


We propose a visual replay module and routing depth scaling to collaboratively enhance visual perception and refine complicated latents for deeper contextual reasoning. The former module leverages causal self-attention to estimate token saliency, reinforcing fine-grained grounding through spatially-coherent constraints. Complementarily, the latter mechanism adaptively allocates additional reasoning steps to complex tokens, enabling deeper contextual refinement.

<div align="center">
  <figure>
    <img src="framework.png" alt="Overview" style="max-width: 100%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of Our Vedas.</em></figcaption>
  </figure>
</div>


## 🔥 News

<div style="max-height: 240px; overflow-y: auto;">

- **[2026.04]** 🎉🎉Initial release of Training Code and Inference Code.

</div>


## 📑 Table of Contents <span id="table-of-contents"></span>

* [🚀 Quick Start](#quick-start)
  * [Installation](#installation)
  * [Data Preparation](#data)
  * [Training](#training)
    * [Qwen2-VL](#qwen2-vl)
    * [Chameleon](#chameleon)
    * [Training Arguments](#arguments)
  * [Inference](#inference)
* [✨ How It Works](#how-it-works)
* [🔗 Related Projects](#related)
* [📚 Citation](#citation)


## 🚀 Quick Start <span id="quick-start"></span>

### 1. Installation <span id="installation"></span>

Clone repo:

```
git clone https://github.com/Simon98-AI/Vedas.git
cd Vedas
```

Setup environment:

```
conda create -n vedas python=3.10 
conda activate vedas
```

Expected folder structure

```plaintext
IVT-LR/
  ├── chameleon
        ├── args/
        ├── chameleon_dataset.py
        ├── ...
  ├── qwen_vl
        ├── args/
        ├── dataset.py
        ├── ...
  └── environment.yml
```

### 2. Data Preparation <span id="data"></span>

Download datasets:

```
dataset = load_dataset("LightChen2333/M3CoT")
dataset = load_dataset("derek-thomas/ScienceQA")
```

or download manually from:

* [M3CoT](https://huggingface.co/datasets/LightChen2333/M3CoT)
* [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)

### 3. Training <span id="training"></span>

> **💡 Skip Training:** If you want to skip training and directly run inference, you can download our pretrained models from the [Vedas Collection]() on Hugging Face.

#### Qwen2-VL <span id="qwen2-vl"></span>

To train the Qwen2-VL model on the M3CoT dataset and the SciceneQA:

```
cd qwen_vl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=disabled
deepspeed --master_port 29505 qwenvl_run_router.py qwen_vl/args/qwen.yaml
    --deepspeed \\
    --deepspeed_config ds_config.json
```

#### Training Arguments <span id="arguments"></span>

Key parameters in configuration:

- `save_path`: Checkpoint save directory
- `name`: Experiment name
- `epochs_per_stage`: Epochs per latent reasoning stage (default: 4)
- `max_latent_stage`: Maximum latent reasoning stages (default: 5)
- `resume`: Resume epoch number (default: 0)
- `batch_size_training`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `num_epochs`: Total training epochs (default: 16)
- `lr`: Learning rate (default: 4e-5)

### 4. Inference <span id="inference"></span>

To generate the answer on the test split, run the inference code.

Qwen2-VL on M3CoT:

```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_mp_m3cot.py > infer.log 2>&1 &  
```

Qwen2-VL on ScienceQA:
```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_mp_scienceqa.py > infer.log 2>&1 &  
```

Qwen2-VL on MMVP:
```
export CUDA_VISIBLE_DEVICES=0
nohup python infer_mp_mmvp.py > infer.log 2>&1 &  
```

```

## 🔗 **Related Projects** <span id="related"></span>

### 📄 Related Papers

- **[Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)**  
  A pioneering work on latent reasoning that uses continuous thought representations for LLM reasoning.

- **[Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space](https://arxiv.org/abs/2510.12603)**   
  A pioneering work on latent reasoning that uses interleaved paradigm for MLLM reasoning.

### 🌟 Awesome Collections

- **[Awesome Latent Space](https://github.com/YU-deep/Awesome-Latent-Space)**  
  A curated collection of resources on latent space methods and applications.

- **[Awesome Latent CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)**  
  A comprehensive list of latent chain-of-thought reasoning resources.


## 📚 **Citation** <span id="citation"></span>

If you use **Vedas** in your research or applications, please consider citing:

```bibtex
@article{han2026vedas,
  title={Visual Enhanced Depth Scaling for Multimodal Latent Reasoning},
  author={Yudong Han, Yong Wang, Zaiquan Yang, Zhen Qu, Liyuan Pan, Xiangxiang Chu},
  journal={arXiv},
  year={2025}
}
```


<br/>
⭐ <b>Thank you for visiting our Vedas!</b> ⭐

</div>
