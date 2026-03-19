# GDCC

This repository contains the code of the paper:

> Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection <br>

In this paper, we propose a generation-detection cycle consistent (GDCC) learning framework that jointly optimizes both layout-to-image (L2I) generation and object detection (OD) tasks in an end-to-end manner. The key of GDCC lies in the inherent duality between the two tasks, where L2I takes all object boxes and labels as input conditions to generate images, and OD maps images back to these layout conditions. Specifically, in GDCC , L2I generation is guided by a layout translation cycle loss, ensuring that the layouts used to generate images align with those predicted from the synthesized images. Similarly, OD benefits from an image translation cycle loss, which enforces consistency between the synthesized images fed into the detector and those generated from predicted layouts. 
![img](./images/overview.png)



## Installation

Clone this repo and create the GDCC environment with conda. We test the code under `python==3.8.13, pytorch==1.12.1, cuda=11.3` on Tesla V100 GPU servers.

1. Initialize the conda environment:

   ```bash
   conda create -n gdcc python=3.8 -y
   conda activate gdcc
   ```

2. Install the required packages:

   ```bash
   cd gdcc
   # when running training
   pip install -r requirements/train.txt
   # only when running inference with DPM-Solver++
   pip install -r requirements/dev.txt
   ```



## Download Pre-trained L2I Generation Models

We provide original L2I generation model and the model fine-tuned with GDCC for comparison. Download and put them into `./pretrained_diffusers/`.

|        Dataset        |  L2I Model   | GDCC Fine-tune | Image Resolution | Grid Size |                           Download                          |
| :-------------------: |:------------:|:--------------:| :--------------: | :-------: | :----------------------------------------------------------: |
|      COCO-Stuff       | GeoDiffusion |       ×        |     256x256      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-256x256) |
|      COCO-Stuff       | GeoDiffusion |       √        |     256x256      |  256x256  | [HF Hub](https://huggingface.co/Strike1999/GeoDiffusion_256x256_GDCC) |
|      COCO-Stuff       | GeoDiffusion |       ×        |     512x512      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-512x512) |
|      COCO-Stuff       | GeoDiffusion |       √        |     512x512      |  256x256  | [HF Hub](https://huggingface.co/Strike1999/GeoDiffusion_512x512_GDCC) |




## Generate images with L2I Generation Models fine-tuning with GDCC

Download the pre-trained models and put them under the root directory. Run the following commands to run detection data generation with GeoDiffusion. For simplicity, we embed the layout definition process in the file `run_layout_to_image.py` directly. Check [here](./run_layout_to_image.py#L75-L82) for detailed definition. Run:

```bash
python run_layout_to_image.py $CKPT_PATH --output_dir ./results/
```

## Evaluate L2I Generation Models
### 1. Generate Images according to COCO annotations
For the 256x256 model:
```bash
bash tools/dist_test.sh
```
For the 512x512 model:
```bash
bash tools/dist_test_512x512.sh
```
**Note:** If you need to compute the **YOLO_Score**, please set `--nsamples` to `5`.
### 2. Evaluate L2I Generation Models based on the generated images
For FID and YOLO Scores, Please refer to [LAMA](https://github.com/ZejianLi/LAMA/tree/main).


## Qualitative Results

More results can be found in the main paper.

![img](./images/qualitative_1.PNG)

![img](./images/qualitative_2.PNG)






## Acknowledgement

We adopt the following open-sourced projects:

- [geodiffusion](https://github.com/KaiChen1998/GeoDiffusion): GeoDiffusion for L2I generation.
- [controlnet](https://github.com/lllyasviel/ControlNet): ControlNet for Controllable generation.
- [controlnet++](https://github.com/xinsir6/https://github.com/liming-ai/ControlNet_Plus_Plus): improve controls with consistency feedback.
- [diffusers](https://github.com/huggingface/diffusers/): basic codebase to train Stable Diffusion models.
- [mmdetection](https://github.com/open-mmlab/mmdetection): dataloader to handle images with various geometric conditions.
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) & [LAMA](https://github.com/ZejianLi/LAMA): data pre-processing of the training datasets.
