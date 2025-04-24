# WateRF: Enhancing Robust Watermarks in Radiance Fields through Crytographic Hashing for Protection of Copyrights (CVPR 2024)
Emmanuel O Ansah(KWAME NKRUMAH SCIENCE AND TECHNOLOGY)

Official Pytorch Implementation of WateRF and Crytographic Hashing.

Paper: [arXiv](https://arxiv.org/abs/2405.02066) | [CVPR 2024 Open Access] LINK TO THE MODIFIED CODE (https://github.com/Naa-Sei/Watermark-and-Cryptographic-Hashingl)<br>
Link to the Original Code: https://kuai-lab.github.io/cvpr2024waterf/

<p align="center">
    <img src="./assets/teaser.png" alt="center" width="70%">
</p>

Abstract: *Neural Radiance Fields (NeRF) are becoming a foundational tool for 3D content creation across diverse domains, yet protecting their copyrights remains an open challenge.The original paper published considered only Watermarking since it is one of the pivotal solutions for safely deploying NeRFbased 3D representationIn. In this work, I proposed a novel framework that fuses digital watermarking with cryptographic hashing to embed robust ownership signatures into NeRF models and their rendered outputs.
The method used fine-tunes NeRFs across both implicit and explicit representations by embedding binary watermark hashes into the frequency domain of the rendered views, specifically leveraging the low-frequency subbands of a Discrete Wavelet Transform (DWT). A pre-trained decoder network, based on Hidden architecture, extracts the embedded signatures reliably.
If a rendered view originates from the protected model, then the cryptographic hash is recovered accurately; else, ownership validation fails.
Optimization proceeds via deferred back-propagation combined with a patch-wise loss, ensuring high bit accuracy while preserving rendering fidelity.
If the fine-tuning embeds the watermark hash effectively, then the model achieves resilience against various distortions, including cropping, compression, and noise, without compromising visual quality.
 This dual-layer watermark–hash fusion ensures both model and rendering-level copyright protection, offering a scalable solution for secure deployment of NeRF-based systems.*

## Installation
First, clone the repository and create a new conda environment:
```bash
git clone https://github.com/kuai-lab/cvpr2024_WateRF.git
cd cvpr2024_WateRF
conda create -n WateRF python=3.9
conda activate WateRF
```
Next, install the required packages:
```bash
pip install torch torchvision  # Make sure to install the appropriate versions for your setup
pip install tqdm scikit-image opencv-python configargparse lpips icecream imageio-ffmpeg kornia tensorboard plyfile pytorch-wavelets pywavelets
```

## Data Preparation
To prepare the dataset and pre-trained weights for training and evaluation, follow these steps:

1. Download the NeRF dataset from [NeRF Datasets Link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

2. Extract the downloaded dataset and place it in the `./data` directory. Your directory structure should look like this:
    ```
    cvpr2024_WateRF/
    ├── data/
    │   ├── NeRF_dataset/
    │   │   ├── llff/
    │   │   ├── syn/
    ├── assets/
    ├── configs/
    ├── train_watermarking_dwt.py
    └── ...
    ```

3. Download the TensoRF pre-trained weights from [TensoRF Pretrained Wieghts link](https://onedrive.live.com/?id=C624178FAB774B7!141&resid=C624178FAB774B7!141&authkey=!AKpIQCzsxSTyFXA&cid=0c624178fab774b7).

4. Place the pre-trained weights in the `./data/TensoRF_weights` directory. Your final directory structure should look like this:
    ```
    cvpr2024_WateRF/
    ├── data/
    │   ├── NeRF_dataset/
    │   │   ├── llff/
    │   │   ├── syn/
    │   ├── TensoRF_weights/
    │   │   ├── weight_file1.th
    │   │   └── weight_file2.th
    ├── assets/
    ├── configs/
    ├── train_watermarking_dwt.py
    └── ...
    ```

5. Download the weights for perceptual loss from [PerceptualSimilarity Github](https://github.com/SteffenCzolbe/PerceptualSimilarity) and place them in the `./loss/losses` directory. Your directory structure should now include:
    ```
    cvpr2024_WateRF/
    ├── data/
    │   ├── NeRF_dataset/
    │   │   ├── llff/
    │   │   ├── syn/
    │   ├── TensoRF_weights/
    │   │   ├── weight_file1.th
    │   │   └── weight_file2.th
    ├── loss/
    │   ├── losses/
    │   │   ├── rgb_watson_vgg_trial0.pth
    │   │   └── ...
    ├── assets/
    ├── configs/
    ├── train_watermarking_dwt.py
    └── ...
    ```

Ensure that the paths in your configuration files are set correctly to match the locations of the dataset, pre-trained weights, and loss weights.

## Usage
### Training
To train the model, run the following command:
```bash
python -u train_watermarking_dwt.py --config configs/lego.txt
```

### Rendering
To render images using the trained model, run:
```bash
python train_watermarking_dwt.py --config configs/lego.txt --ckpt path/to/your/watermarked_checkpoint --render_only 1 --render_test 1
```

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{jang2024waterf,
  title={WateRF: Robust Watermarks in Radiance Fields for Protection of Copyrights},
  author={Jang, Youngdong and Lee, Dong In and Jang, MinHyuk and Kim, Jong Wook and Yang, Feng and Kim, Sangpil},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12087--12097},
  year={2024}
}
```

## TO-DO List
- [ ] Load watermarked weights and result
- [ ] Add instructions for using a custom dataset
# Enhancing-Radiance-Field-Watermarking-Robustness-through-Cryptographic-Hashing-Technique
