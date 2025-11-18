<div align="center">
<h1>
    <a href="https://arxiv.org/abs/2506.09538v2" target="_blank">
        AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches (NeurIPS 2025)
    </a>
</h1>

<img src="./pages/static/images/fig1.png" width="800">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

<br>

Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce **Angle-Robust Concept Learning (AngleRoCL)**, a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with **over 50% average relative improvement** in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents.

<br>

<div align="center">
<img src="./pages/static/images/fig2.jpg" width="700">
</div>

## 💥 News 💥
- **`24.10.2025`** | Accepted by NeurIPS 2025! 🎉
- **`14.06.2025`** | Paper available on arXiv: [Link](https://arxiv.org/abs/2506.09538v2)

# Getting started

## Requirements
The code requires Python 3.10.16 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.

```bash
# Create conda environment
conda create -n anglerocl python=3.10.16
conda activate anglerocl

# Install PyTorch (tested on RTX 3090/4090)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install safetensors==0.5.3
pip install transformers==4.51.3
pip install accelerate==1.4.0
pip install diffusers==0.32.2
pip install opencv-python==4.8.1.78
pip install pandas==2.2.3
pip install kornia==0.6.8
pip install numpy==1.23.1
pip install scikit-learn==1.3.1

# Install MMDetection (for multi-detector evaluation)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## Pretrained Models and Datasets

### Required Downloads

1. **Stable Diffusion v1.5** - Download from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

2. **Our Pretrained Resources** - [Google Drive](https://drive.google.com/drive/folders/1CaFqm3_LqxST0e74xxDxuMywqdDlW8Xs?usp=sharing)
   - [angle-robust.safetensors](https://drive.google.com/file/d/1avOTyBqqcrrA6TTCJD1o2CN_URtnwHwB/view?usp=drive_link) - Trained angle-robust concept embedding
   - [Generated patches](https://drive.google.com/file/d/1oS_Urh7BVLAh5HO2UFOwbHuwsqqA6cT4/view?usp=drive_link) - Pre-generated adversarial patches
   - [Detector checkpoints](https://drive.google.com/drive/folders/1EkXVjcGJiYeZtiYqf4FAqekeJ-M3XMHi?usp=sharing) - YOLOv3/v5/v10, DETR, Faster R-CNN, RT-DETR weights

## Resources
The code was tested on NVIDIA RTX 3090/4090 GPUs with 24GB VRAM.

## Training Angle-Robust Concept
You can train the angle-robust concept using the command below:
```bash
accelerate launch anglerocl.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --placeholder_token="<angle-robust>" \
  --initializer_token="robust" \
  --output_dir="runs/anglerocl/${timestamp}" \
  --yolo_weights_file="yolov5s.pt" \
  --max_train_steps=50000 \
  --save_steps=1950 \
  --validation_steps=1950
```
- `pretrained_model_name_or_path` path to pre-trained Stable Diffusion model
- `placeholder_token` the concept token to learn (default: "<angle-robust>")
- `initializer_token` token to initialize the concept (default: "robust")
- `output_dir` where the learned embeddings will be saved (${timestamp} will be replaced automatically)
- `yolo_weights_file` path to YOLOv5 detector checkpoint
- `max_train_steps` total training steps (default: 195000, recommended: 50000 for faster training)
- `save_steps` save embeddings every N steps (default: 1950)
- `validation_steps` run validation every N steps (default: 1950)


## Generating Adversarial Patches
After training the angle-robust concept, you can generate adversarial patches using the learned embedding.

### Generate Dataset
You can generate datasets using the scripts in the `dataset/` folder:
```bash
# Generate NDDA baseline dataset
python dataset/NDDA.py \
  --output-dir=<output_path> \
  --num-images=50

# Generate NDDA + AngleRoCL dataset
python dataset/NDDA_textual_inversion.py \
  --output-dir=<output_path> \
  --num-images=50

# Generate with prompt tuning
python dataset/NDDA_tuneprompt.py \
  --output-dir=<output_path> \
  --num-images=50 \
  --group="all"
```

### Generate Single Image
You can also generate a single image using `generate.py`:
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Load the trained angle-robust concept
repo_id_embeds = "<path_to_learned_embeds>"
pipe.load_textual_inversion(repo_id_embeds)

# Generate image with angle-robust concept
prompt = "A <angle-robust> blue stop sign with 'abcd' on it"
image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("angle_robust_stop_sign.png")
```

## Multi-View Testing
You can evaluate the generated patches using the testing scripts in the `test/` folder.

There are **four testing scripts** organized in two groups:

### Group 1: Pure Background Testing (No Environment Images)

```bash
# Process all textures in a single folder
python test/multiview_detection_folder_multidetector.py \
  --texture-folder= \
  --detector=yolov5 \
  --yolov5-model= \
  --target-class-id=11 \
  --output-dir=

# Batch process multiple folders (e.g., different prompt categories)
python test/multiview_detection_folder_batch_multidetector.py \
  --base-dir= \
  --detector=yolov5 \
  --yolov5-model= \
  --target-class-id=11 \
  --output-base-dir=
```

### Group 2: Physical Environment Testing (With Background Images)

```bash
# Process all textures in a single folder with environment backgrounds
python test/multiview_detection_folder_multidetector_environment.py \
  --texture-folder= \
  --environment-folder= \
  --detector=yolov5 \
  --yolov5-model= \
  --target-class-id=11 \
  --output-dir=

# Batch process multiple folders with environment backgrounds
python test/multiview_detection_folder_batch_multidetector_environment.py \
  --base-dir= \
  --environment-folder= \
  --detector=yolov5 \
  --yolov5-model= \
  --output-base-dir=
```

### Key Differences:
- **`_folder`** scripts: Process **all patches within one folder**
- **`_folder_batch`** scripts: Process **all patches across multiple subfolders** (batch mode with category grouping)
- **Without `_environment`**: Test on pure color backgrounds
- **With `_environment`**: Test on real-world environment backgrounds

**Example Directory Structure:**
```
# For _folder scripts:
patches/
├── image_001.png
├── image_002.png
└── image_003.png

# For _folder_batch scripts:
all_patches/
├── blue_square_stop_sign/
│   ├── image_001.png
│   └── image_002.png
├── stop_sign_with_hello/
│   ├── image_001.png
│   └── image_002.png
└── yellow_triangle_stop_sign/
    ├── image_001.png
    └── image_002.png
```

**Output files:**
- `angle_confidence.csv` - detection confidence at each angle
- `confidence_curve.png` - visualization of angle-confidence curve  
- `aasr_analysis/` - detailed AASR metrics and analysis
- `summary_results.csv` - (for `_folder_batch` scripts) aggregated results and category-wise statistics


## Acknowledgement

We thank the authors of the following outstanding open-source repositories for their valuable code and contributions:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) – the foundation of our text-to-image pipeline  
- [P2P: Prompt-to-Perturb](https://github.com/yasamin-med/P2P/tree/main) – pioneering work on text-guided adversarial attacks that greatly inspired this project (CVPR 2025)  
- [yolov5_adversarial](https://github.com/SamSamhuns/yolov5_adversarial) – excellent reference implementation for physical-world adversarial patch attacks on YOLO detectors  

## Citation
```
@article{ji2025anglerocl,
  title={AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches},
  author={Ji, Wenjun and Fu, Yuxiang and Ying, Luyang and Fan, Deng-Ping and Wang, Yuyi and Cheng, Ming-Ming and Tsang, Ivor and Guo, Qing},
  journal={arXiv preprint arXiv:2506.09538},
  year={2025}
}
```
