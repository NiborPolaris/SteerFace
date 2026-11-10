# SteerFace

<a href="https://dl.acm.org/doi/10.1145/3767308.3835280"><img src="https://img.shields.io/badge/ACM MM-2026-gold" alt="ACM MM 2026"></a>
<a href="https://arxiv.org/abs/2605.30894"><img src="https://img.shields.io/badge/arXiv-2605.30894-b31b1b.svg" alt="arXiv"></a>

Official repository for the ACM Multimedia 2026 paper, **SteerFace: Debiasing Synthetic Face Generation via Adaptive Residue Perturbation**.

<img width="1734" height="567" alt="SteerFace overview" src="https://github.com/user-attachments/assets/12c1e052-110a-4412-b9f5-a5793783e3dc" />

SteerFace trains an identity-conditioned latent diffusion model with adaptive residue perturbation. At inference, a fixed identity embedding conditions ordinary diffusion sampling; no identity perturbation is applied.

This repository supports four workflows:

1. Train the SteerFace generator from scratch.
2. Generate images with our pretrained generator.
3. Train a face recognition (FR) model on our released synthetic images.
4. Evaluate our pretrained FR model using the original TFace evaluation tools.

## Resources

| Resource | Download |
| --- | --- |
| Pretrained VAE encoder / decoder | [Google Drive](https://drive.google.com/drive/folders/1d-zs3yjsnzOMNkz7qy3JSb-fMf0UmSdT?usp=sharing) ([IDiff-Face](https://github.com/fdbtrs/IDiff-Face)) |
| Pretrained SteerFace generator | [Hugging Face](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_gen_model_250k.pth) |
| CASIA-WebFace dataset | [Kaggle](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) |
| Released synthetic dataset | [Hugging Face](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_gen_samples.tar.gz) |
| Pretrained FR model | [Hugging Face](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_fr_model.pth) |
| 10K identity reference images | [Hugging Face](https://huggingface.co/datasets/LuxPolaris/MorphFace/blob/main/source_10k_img.tar.gz) ([MorphFace](https://huggingface.co/datasets/LuxPolaris/MorphFace)) |
| Embeddings of 10K references | [Hugging Face](https://huggingface.co/datasets/LuxPolaris/MorphFace/blob/main/source_10k_emb.pth) ([MorphFace](https://huggingface.co/datasets/LuxPolaris/MorphFace)) |
| Pretrained ElasticFace backbone | [Google Drive](https://drive.google.com/drive/folders/1q3ws_BQLmgXyiy2msvHummXq4pRqc1rx) ([ElasticFace](https://github.com/fdbtrs/ElasticFace)) |


## Environment

Install from the repository root:

```bash
pip install torch torchvision
pip install -r requirements.txt
pip install -r generative_model_training/requirements.txt -c constraints-generation.txt
```

For FR training:

```bash
pip install -r recognition_model_training/requirements.txt
```

For image-to-TFRecord conversion, also install:

```bash
pip install -r recognition_model_training/requirements-conversion.txt
```

## Directory layout

```text
SteerFace/
├── requirements.txt
├── dataset/
│   ├── CASIA/
│   │   ├── images/<identity>/<image>.jpg
│   │   └── elasticface_embeddings/
│   │       ├── elastic_embed.npy
│   │       └── image_names.pkl
│   ├── context_database/source_10k_emb.pth
│   ├── synthetic/images/<integer_identity>/<image>.png
│   └── synthetic/TFR_training_data/
├── generative_model_training/
│   ├── ckpt/
│   │   ├── autoencoder/
│   │   │   ├── first_stage_encoder_state_dict.pt
│   │   │   └── first_stage_decoder_state_dict.pt
│   │   ├── elasticface/<downloaded_backbone>.pth
│   │   └── steerface/steerface_gen_model_250k.pth
│   ├── extract_embeddings.py
│   ├── main.py
│   └── sample.py
└── recognition_model_training/
    ├── train.yaml
    ├── tface_train.py
    └── ckpt/steerface_fr_model.pth
```

## 1. Train SteerFace generator from scratch

### Prepare data and models

Download [CASIA-WebFace](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface), the [ElasticFace backbone](https://drive.google.com/drive/folders/1q3ws_BQLmgXyiy2msvHummXq4pRqc1rx), and the [VAE encoder/decoder](https://drive.google.com/drive/folders/1d-zs3yjsnzOMNkz7qy3JSb-fMf0UmSdT?usp=sharing). Place the training images under `dataset/CASIA/images/`.

### Extract training identity embeddings

Run from `generative_model_training/`:

```bash
cd generative_model_training
python extract_embeddings.py \
  --images ../dataset/CASIA/images \
  --checkpoint ckpt/elasticface/YOUR_BACKBONE.pth \
  --output ../dataset/CASIA/elasticface_embeddings/elastic_embed.npy \
  --names ../dataset/CASIA/elasticface_embeddings/image_names.pkl \
  --batch-size 512 \
  --device cuda:0
```

### Start training

Run from `generative_model_training/`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py
```

Train for **250,000 steps** with Adam, learning rate `1e-4`, and total batch size 512 (64 per GPU on 8 GPUs).

To use custom paths:

```bash
python main.py \
  paths.data_root=/absolute/path/to/dataset \
  training.checkpoint.VQEncoder=/absolute/path/to/encoder.pt \
  training.checkpoint.VQDecoder=/absolute/path/to/decoder.pt \
  hydra.run.dir=/absolute/path/to/training_run
```

Checkpoints are saved inside the run directory's `checkpoints/` subdirectory. EMA checkpoint names follow `ema_averaged_model_<step>.ckpt`, including `ema_averaged_model_250000.ckpt`.

## 2. Generate synthetic images

To reuse our pretrained SteerFace generator, download the [250K generator](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_gen_model_250k.pth), [VAE encoder/decoder](https://drive.google.com/drive/folders/1d-zs3yjsnzOMNkz7qy3JSb-fMf0UmSdT?usp=sharing), and [precomputed 10K identity embeddings](https://huggingface.co/datasets/LuxPolaris/MorphFace/blob/main/source_10k_emb.pth).

From `generative_model_training/`:

```bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
  checkpoint.path=./ckpt/steerface/steerface_gen_model_250k.pth \
  sampling.contexts_file=../dataset/context_database/source_10k_emb.pth \
  sampling.save_dir=./outputs/steerface_samples \
  sampling.n_contexts=10000 \
  sampling.n_samples_per_context=50 \
  sampling.batch_size=50
```

If the autoencoder weights are stored elsewhere, append:

```bash
autoencoder.encoder=/absolute/path/to/encoder.pt \
autoencoder.decoder=/absolute/path/to/decoder.pt
```

The default settings generate **10,000 identities × 50 images**, saved at 112 × 112:

```text
outputs/steerface_samples/
├── 0/0.png ... 49.png
├── 1/0.png ... 49.png
└── ...
```

Completed identity folders are skipped when sampling resumes.

## 3. Train FR model

### Download and extract the released synthetic dataset

Download [steerface_gen_samples.tar.gz](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_gen_samples.tar.gz) and extract it. Set the image root to the directory containing the integer identity folders:

```text
dataset/synthetic/images/
├── 0/
├── 1/
└── ...
```

### Convert images to TFRecords

From `recognition_model_training/`:

```bash
cd recognition_model_training
python tools/img2tfrecord.py \
  --img_list ../dataset/synthetic/images \
  --tfrecords_dir ../dataset/synthetic/TFR_training_data \
  --tfrecords_name Syn_10k
```

The converter writes `Syn_10k.index` and TFRecord shards to the output directory.

The converter includes all images in each identity folder.

### Configure and launch FR training

Set these fields in `train.yaml`:

```yaml
DATA_ROOT: '../dataset/synthetic/TFR_training_data'
INDEX_ROOT: '../dataset/synthetic/TFR_training_data'
DATASETS:
  - name: Syn_10k
    batch_size: 64
    weight: 1.0
    scale: 64
    margin: 0.5
BACKBONE_NAME: 'IR_50'
HEAD_NAME: 'ArcFace'
METHOD: 'TF-Synthetic'
NUM_EPOCH: 32
SAVE_EPOCHS: [22, 28, 32]
STAGES: [22, 28]
```

Launch training on 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
  tface_train.py --config train.yaml
```

Training runs for **32 epochs**, with learning-rate decay at epochs 22 and 28. Probabilistic random-resized-crop augmentation is controlled by `AdaFace_augment_prob` (default 0.2). When changing GPU count, update `WORLD_SIZE`, launch settings, and per-device batch size consistently.

## 4. Evaluation

Download [steerface_fr_model.pth](https://huggingface.co/datasets/LuxPolaris/SteerFace/blob/main/steerface_fr_model.pth), our pretrained **IR-50** model.

Follow the original [TFace evaluation instructions](https://github.com/Tencent/TFace/tree/master/recognition/test) for test data preparation and evaluation.

For the five verification benchmarks, prepare `lfw.bin`, `cfp_fp.bin`, `agedb_30.bin`, `cplfw.bin`, and `calfw.bin` as required by TFace, then run in a separate TFace checkout:

```bash
git clone https://github.com/Tencent/TFace.git
cd TFace/recognition/test
CUDA_VISIBLE_DEVICES=0 python verification.py \
  --ckpt_path=/absolute/path/to/steerface_fr_model.pth \
  --data_root=/absolute/path/to/validation_bins \
  --backbone=IR_50
```

## Citation

If you use this repository or resources from our papers, please cite us:

```bibtex
@inproceedings{mi2026steerface,
  title = {SteerFace: Debiasing Synthetic Face Generation via Adaptive Residue Perturbation},
  author = {Mi, Yuxi and Yuan, Qiuyang and Xu, Jianqing and Zhou, Yichun and Zhao, Xuan and Wang, Jun and Guo, Rizen and Zhou, Shuigeng},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  pages = {7756--7765},
  year = {2026},
  doi = {10.1145/3767308.3835280}
}

@inproceedings{mi2025morphface,
  title = {Data synthesis with diverse styles for face recognition via 3dmm-guided diffusion},
  author = {Mi, Yuxi and Zhong, Zhizhou and Huang, Yuge and Yuan, Qiuyang and Zhao, Xuan and Xu, Jianqing and Ding, Shouhong and Wang, Shaoming and Guo, Rizen and Zhou, Shuigeng},
  booktitle = {2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {21203--21214},
  year = {2025},
  organization = {IEEE}
}
```

## Acknowledgements

Built on [IDiff-Face](https://github.com/fdbtrs/IDiff-Face), [ElasticFace](https://github.com/fdbtrs/ElasticFace), [MorphFace](https://huggingface.co/datasets/LuxPolaris/MorphFace), [TFace](https://github.com/Tencent/TFace), and [DCFace](https://github.com/mk-minchul/dcface). We thank the authors for sharing their code and resources.
