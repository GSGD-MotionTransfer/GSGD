<h1 align="center">Gaussian See, Gaussian Do</h1>
<h2 align="center">3D Semantic Motion Transfer</h2>

<p align="center">
  <strong>SIGGRAPH Asia 2025</strong>
</p>

<p align="center">
  <a href="https://gsgd-motiontransfer.github.io/">
    <img src="assets/logo_page.svg" alt="Project Page" width="125">
  </a>
  <a href="https://arxiv.org/abs/2511.14848">
    <img src="assets/logo_arxiv.svg" alt="arXiv" width="125">
  </a>
  <a href="https://arxiv.org/pdf/2511.14848.pdf">
    <img src="assets/logo_paper.svg" alt="Paper" width="125">
  </a>
</p>

<p align="center">
  <strong>Authors:</strong> Yarin Bekor<sup>1,*</sup>, Gal Harari<sup>1,*</sup>, Or Perel<sup>2,3,4</sup>, Or Litany<sup>1,2</sup><br>
  <sup>1</sup>Technion, <sup>2</sup>NVIDIA, <sup>3</sup>University of Toronto, <sup>4</sup>Vector Institute<br>
  <sup>*</sup>Indicates Equal Contribution
</p>

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="800">
</p>

---

## Overview

**Gaussian See, Gaussian Do** is a novel approach for semantic 3D motion transfer from multiview video. Our method enables rig-free, cross-category motion transfer between objects with semantically meaningful correspondence. Building on implicit motion transfer techniques, we extract motion embeddings from source videos via condition inversion, apply them to rendered frames of static target shapes, and use the resulting videos to supervise dynamic 3D Gaussian Splatting reconstruction.

---

## Quick Downloads (Required Assets)

Before running the examples, download and extract the following:

### 1. Trained Motion Embeddings

Download the [Motion Embeddings](https://drive.google.com/file/d/1rbCD95gBu7S9f33Zn8hVwDCvi3fQ8IlC/view?usp=sharing) and extract to `./Outputs/MotionEmbeddings/`. These are pre-inverted motions that allow you to skip multi-hour inversion and immediately transfer motions to a new target.

**Download via terminal:**
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1rbCD95gBu7S9f33Zn8hVwDCvi3fQ8IlC -O pretrained_motion_embeddings.zip
mkdir -p ./Outputs/MotionEmbeddings/
unzip -q pretrained_motion_embeddings.zip -d ./Outputs/MotionEmbeddings/ && rm pretrained_motion_embeddings.zip
```

### 2. `web_crawled` Dataset

Download the [Web Crawled Dataset](https://drive.google.com/file/d/1nlaF3_2GvnWonP3b_zZftTR3LQGwPc0n/view?usp=sharing) and extract to `./data/web_crawled/`. It contains:
- Multiview motion videos under `motions/` (organized by category, e.g., `motions/animal/horse_riggedgame_ready/`)
- 3DGS targets (with prompts and optional first frames) under `targets/` (organized by category, e.g., `targets/human_like_objects/basic_crop_top_and_pants.ply`)

**Download via terminal:**
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1nlaF3_2GvnWonP3b_zZftTR3LQGwPc0n -O gsgd_web_crawled_dataset.zip
mkdir -p ./data/web_crawled/
unzip -q gsgd_web_crawled_dataset.zip -d ./data/web_crawled/ && rm gsgd_web_crawled_dataset.zip
```

Place both downloads at the repository root so the paths match the examples below.

### 3. Mini-Mixamo Dataset (Optional) - **TODO**

**Note**: This section is not yet published. Instructions for the Mixamo dataset will be added here.

This dataset enables evaluation with supervised metrics and provides human motions that compose the main part of the `web_crawled` human motions. The Mixamo motions are not included in the downloadable `web_crawled` dataset because Mixamo's license requires downloading the data manually after registration.

**TODO**: To create the mini-Mixamo dataset:
- First, manually download the Mixamo data as described in the [Mixamo download instructions](#mixamo-download-instructions) section below
- **TODO**: Then, use the `prepare_datasets.sh` script to render it into the correct format:
  ```bash
  bash prepare_datasets.sh
  ```
- This will add the missing human motions to your `web_crawled` dataset

**Note**: This step is optional but recommended if you want to:
- Evaluate results with supervised metrics (requires ground truth)
- Use the full set of human motions available in the web_crawled dataset

---

## Setup

### Clone the Repository

```bash
git clone --recursive git@github.com:YarinBekor/MotionTransfer3D.git
cd MotionTransfer3D
```

### Create the Conda Environment

```bash
source setup_MT3D.sh
```

This will create and activate the `mt3d` conda environment with all required dependencies.

---

## Quick Start Examples

### Example 1: Apply a Pre-trained Motion Embedding to a Target

This is the fastest path: reuse a pre-trained motion embedding and transfer it onto a target 3DGS.

```bash
conda activate mt3d

python pipeline_given_embedding.py \
  --config configs/MT3D.yaml \
  --target_path data/web_crawled/targets/human_like_objects/basic_crop_top_and_pants.ply \
  --motion_embedding_path Outputs/MotionEmbeddings/Human_BreakdanceReady \
  --output_path quickstart_breakdance_on_basicpants
```

Add `--run_evaluation` to execute evaluation at the end (requires source videos to compute motion metrics).

**Outputs:**
- First frames: `./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/FirstFrames/`
- Supervision gifs: `./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/Supervision/`
- Final reconstruction: `./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/Final_Output/final_output/`

---

### Example 2: Motion Inversion (Train Motion Embeddings)

If you need to invert a new motion (learn motion embeddings) from source videos:

```bash
conda activate mt3d

python train_mlp_reenact.py \
  --config configs/MT3D.yaml \
  --output_path my_horse_motion \
  --source_path data/web_crawled/motions/animal/horse_riggedgame_ready \
  --num_anchors 5
```

The learned embeddings will be saved in `./Outputs/MotionEmbeddings/my_horse_motion/` (one subfolder per view angle).

---

### Example 3: Evaluate Results

Evaluate the reconstructed output against source motion videos.

**First, set up the evaluation environment:**
```bash
source setup_cotracker.sh
```

Then run the evaluation:
```bash
conda activate cotracker

python evaluate_output.py \
  --config configs/MT3D.yaml \
  --output_path ./Outputs/z_Evaluations/quickstart_breakdance_on_basicpants \
  --pred_dir ./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/Final_Output/final_output \
  --source_dir data/web_crawled/motions/human_worked_well/Brian_BreakdanceReady \
  --target_views ./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/FirstFrames \
  --prompt "$(cat data/web_crawled/targets/human_like_objects/basic_crop_top_and_pants.txt)"
```

**Note:** The `--source_dir` argument is optional but recommended for motion fidelity metrics. If you don't have the source videos, omit this argument.

**Output Location:** `./Outputs/z_Evaluations/quickstart_breakdance_on_basicpants/Human_BreakdanceReady_basic_crop_top_and_pants/`

**Result:** Evaluation metrics and comparison results.

---

## How to Use Your Own Target

**Important:** Your 3DGS model must have no spherical harmonics. When creating the 3DGS using the [gaussian-splatting repository](https://github.com/graphdeco-inria/gaussian-splatting), use `--sh_degree 0`. (For example:
```bash
python 3dgs/train.py -s "$dataset_path/dataset" --model_path "$dataset_path/output" --disable_viewer --sh_degree 0
```
)\
\
To use your own 3D Gaussian Splatting (3DGS) model as a target, you need to prepare it in the same format as the targets in the `web_crawled` dataset. Specifically, you need to manually scale and rotate your target 3DGS to match the coordinate system and scale used in our pipeline.



We recommend using the [SuperSplat editor](https://superspl.at/editor) for this normalization process:

1. **Import your model and a reference**: Open the SuperSplat editor and import both:
   - Your target `.ply` file
   - One of the web-crawled target `.ply` files as a reference (e.g., `data/web_crawled/targets/human_like_objects/basic_crop_top_and_pants.ply`)

2. **Select your model**: Mark/select your `.ply` file in the editor

3. **Apply transformations**: Use the transform tools to adjust your model:
   - **Rotation**: Rotate your model to match the orientation of the reference
   - **Scale**: Scale your model to match the size of the reference

4. **Export the normalized model**: 
   - Hide the reference model (mark with the eye icon so only your normalized model is visible)
   - Use **File → Export as PLY** to save your normalized target

5. **Use the normalized model**: Use the exported `.ply` file as your `--target_path` in the pipeline examples above.

**Note**: The normalization step is important for the pipeline to work correctly, as it ensures your target matches the expected coordinate system and scale.

---

## Tips

### Motion Inversion and Reenactment Behavior

Motions that work well on one target tend to work well on many targets. Motions that fail strongly on a target will likely fail for all targets. This is because the motion inversion process usually works or doesn't work (it's target-independent), and motion reenactment given the inverted motion is typically not sensitive to the exact target view.

### Fixing Target Deformation Issues

If the target slightly breaks or deforms incorrectly during reconstruction, you can increase the ARAP (As-Rigid-As-Possible) loss weight to enforce better shape preservation:

```bash
conda activate mt3d

python consolidate_4d.py \
  --config configs/MT3D.yaml \
  --target_path data/web_crawled/targets/human_like_objects/basic_crop_top_and_pants.ply \
  --specify_supervision ./Outputs/Human_BreakdanceReady/basic_crop_top_and_pants/quickstart_breakdance_on_basicpants/Supervision \
  --motion_embedding_path Outputs/MotionEmbeddings/Human_BreakdanceReady \
  --lambda_arap 5.0
```

The default `lambda_arap` is `1.0`. Try increasing it to `2.0`, `5.0`, or `10.0` if you see deformation issues. Higher values enforce stronger shape preservation but may reduce motion flexibility.

---

## Results Location Summary

After running the examples, results are stored in the following locations:

1. **Motion Embeddings:** `./Outputs/MotionEmbeddings/{output_path}/`
   - Contains motion embedding files (`.pt`) including anchor embeddings
   - Contains MLP model checkpoints if using `train_mlp_reenact.py`

2. **First Frames:** `./Outputs/{source_name}/{target_name}/{output_path}/FirstFrames/`
   - Contains PNG images: `0.0.png`, `22.5.png`, etc.
   - One image per viewing angle

3. **Supervision Videos:** `./Outputs/{source_name}/{target_name}/{output_path}/Supervision/`
   - Contains GIF files: `inference_0.0.gif`, `inference_22.5.gif`, etc.
   - May also contain subdirectories with frame-by-frame PNGs

4. **Final Reconstruction:** `./Outputs/{source_name}/{target_name}/{output_path}/Final_Output/`
   - `final_output/`: Contains reconstructed GIFs (e.g., `step_0.gif`, `step_1.gif`)
   - `Checkpoints/`: Contains training checkpoints for the 3D reconstruction

5. **Evaluation Results:** `./Outputs/z_Evaluations/{output_path}/{source_name}_{target_name}/`
   - Contains evaluation metrics and comparison results
   - Includes tracking results and quality metrics

**Note:** All `Outputs/` directories are created relative to the current working directory (the repository root) when scripts run. The paths above are relative to where you execute the scripts.

---

## Mixamo Download Instructions - **TODO**

<a name="mixamo-download-instructions"></a>

**Note**: This section is not yet published. Instructions for manually downloading Mixamo data will be added here. This includes:
- Creating a Mixamo account and registration steps
- Which motions to download
- Where to place the downloaded files
- Format requirements

**TODO**: Once you have downloaded the Mixamo data, follow the steps in the "Quick downloads" section above to use `prepare_datasets.sh` to process it.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{bekor2025gaussian,
  title={Gaussian See, Gaussian Do: Semantic 3D Motion Transfer from Multiview Video},
  author={Bekor, Yarin and Harari, Gal Michael and Perel, Or and Litany, Or},
  journal={arXiv preprint arXiv:2511.14848},
  year={2025}
}
```

---

## License

This project is provided for research purposes. Please refer to the LICENSE file for details.

---

## Acknowledgments

We thank Manuel Kansy and Uriel Singer for helpful discussions. Or Litany acknowledges the support of the Israel Science Foundation (grant No. 624/25) and the Azrieli Foundation Early Career Faculty Fellowship.
