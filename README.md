# Torch-KWT
PyTorch implementation of the paper, [*Keyword Transformer: A Self-Attention Model for Keyword Spotting*](https://arxiv.org/abs/2104.00769). </br>
Made in **Linux OS (WSL)**


## Explanation of Transformer Model for Keyword Spotting
The Transformer architecture has been successful across many domains, including natural language processing, computer vision, and speech recognition. In keyword spotting, self-attention has primarily been used on top of convolutional or recurrent encoders. We investigate various ways to adapt the Transformer architecture to keyword spotting and introduce the Keyword Transformer (KWT), a fully self-attentional architecture that exceeds state-of-the-art performance across multiple tasks without any pre-training or additional data. Surprisingly, this simple architecture outperforms more complex models that mix convolutional, recurrent, and attentive layers. KWT can be used as a drop-in replacement for these models, setting two new benchmark records on the Google Speech Commands dataset with 98.6% and 97.7% accuracy on the 12 and 35-command tasks, respectively.

## Figure 1
![Figure 1: The Keyword Transformer architecture](path/to/your/figure.png)
*Figure 1: The Keyword Transformer architecture. Audio is preprocessed into a mel-scale spectrogram, which is partitioned into non-overlapping patches in the time domain. Together with a learned class token, these form the input tokens for a multilayer Transformer encoder. As with ViT [2], a learned position embedding is added to each token. The output of the class token is passed through a linear head and used to make the final class prediction.*

## Model Working and Techniques Used

### 2.1 Keyword Spotting 
Keyword spotting is used to detect specific words from a stream of audio, typically in a low-power always-on setting such as smart speakers and mobile phones. To achieve this, audio is processed locally on the device. In addition to detecting target words, classifiers may also distinguish between “silence” and “unknown” for words or sounds that are not in the target list. 

In recent years, machine learning techniques, such as deep neural networks (DNN), convolutional neural networks (CNN), recurrent neural networks (RNN), and HybridTree neural networks, have proven to be useful for keyword spotting. These networks are typically used with a preprocessing pipeline that extracts the mel-frequency cepstrum coefficients (MFCC). 

Zhang et al. investigated several small-scale network architectures and identified depthwise-separable CNN (DS-CNN) as providing the best classification/accuracy tradeoff for memory footprint and computational resources. Other works have improved upon this result using synthesized data, temporal convolutions, and self-attention. Recently, Rybakov et al. achieved a new state-of-the-art result on Google Speech Commands using MHAtt-RNN, a non-streaming CNN, RNN, and multi-headed self-attention model.

### 2.2 Self-Attention and the Vision Transformer
Dosovitskiy et al. introduced the Vision Transformer (ViT) and showed that Transformers can learn high-level image features by computing self-attention between different image patches. This simple approach outperformed CNNs but required pre-training on large datasets. 

While Transformers have been explored for wake word detection and voice triggering, fully-attentional models based on the Transformer architecture have not been investigated for keyword spotting. Our approach is inspired by ViT, in that we use patches of the audio spectrogram as input and closely follow recent findings to understand how generally this technique applies to new domains.

### 3 The Keyword Transformer
#### 3.1 Model Architecture
Let \(X ∈ RT× F\) denote the output of the MFCC spectrogram, with time windows \(t = 1,...,T\) and frequencies \(f = 1,...,F\). The spectrogram is first mapped to a higher dimension \(d\) using a linear projection matrix \(W_0 ∈ RF× d\) in the frequency domain. 

A learnable class embedding \(X_{class} ∈ R^{1× d}\) is concatenated with the input in the time-domain. Then, a learnable positional embedding matrix \(X_{pos} ∈ R^{(T+1)× d}\) is added, such that the input representation fed into the Transformer encoder is given by:
\[
X_0 =[X_{class};XW_0] + X_{pos}
\]

The projected frequency-domain features are then fed into a sequential Transformer encoder consisting of \(L\) multi-head attention (MSA) and multi-layer perceptron (MLP) blocks. 

The model size can be adjusted by tuning the parameters of the Transformer. Following recent works, we fix the number of sequential Transformer encoder blocks to 12 and let \(d/k = 64\), where \(d\) is the embedding dimension and \(k\) is the number of attention heads. By varying the number of heads as \(k = 1,2,3\), we end up with three different models.

#### 3.2 Knowledge Distillation
Knowledge distillation uses a pre-trained teacher’s predictions to provide an auxiliary loss to the student model being trained. We apply this technique to improve the performance of our Transformer models.

--- 


## Setup

```bash
git clone <https://github.com/RamakrishnaReddyPalle/Keyword-Transformer-Model-ASR-Free-KWS.git>
```
```bash
cd path\to\cloned\folder
```
```bash
pip install -r requirements.txt
```

## Dataset
To download the Google Speech Commands V2 dataset, you may run the provided bash/WSL (Windows Subsystem for Linux) script as below. This would download the dataset to the "destination path" provided (preferably a folder named "data" inside this root directory).

```bash
sh ./download_gspeech_v2.sh <destination_path>
```

## Training
The Speech Commands V2 dataset provides a "validation_list.txt" file and a "testing_list.txt" file. Run:

```bash
python make_data_list.py -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

This will create the files `training_list.txt`, `validation_list.txt`, `testing_list.txt`, and `label_map.json` at the specified output directory.

Running `train.py` is fairly straightforward. Only a path to a config file is required. Inside the config file, you'll need to add the paths to the .txt files and the label_map.json file created above.

```bash
python train.py --conf path/to/config.yaml
```

Refer to the [example config](sample_configs/base_config.yaml) to see how the config file looks like.

## Inference
You can use the pre-trained model (or a model you trained) for inference using the following scripts:

- `inference.py`: For short ~1s clips, like the audios in the Speech Commands dataset.
- `window_inference.py`: For running inference on longer audio clips, where multiple keywords may be present. Runs inference on the audio in a sliding window manner.

```bash
python inference.py --conf sample_configs/base_config.yaml \
                    --ckpt <path to pretrained_model.ckpt> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device cpu \
                    --batch_size 8   # should be possible to use much larger batches if necessary, like 128, 256, 512 etc.

!python window_inference.py --conf sample_configs/base_config.yaml \
                    --ckpt <path to pretrained_model.ckpt> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device cpu \
                    --wlen 1 \
                    --stride 0.5 \
                    --thresh 0.85 \
                    --mode multi
```
**For a detailed usage of all the scripts for inference, check the `Prediction.ipynb`.**


---

# Symbol Spotting on Digital Architectural Floor Plans

This project implements a deep learning-based framework for symbol spotting on digital architectural floor plans. The primary goal is to identify specific symbols in architectural drawings using advanced image processing techniques.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Input Data](#input-data)
- [Model Outputs](#model-outputs)
- [Conclusion](#conclusion)

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the model and perform symbol spotting, execute the following command:

```bash
python main.py
```

Ensure you have your input data prepared as specified in the `data/` directory.

## Model Architecture

<!-- Add model architecture figure here -->

## Input Data

<!-- Add screenshots of input data for inference here -->

## Model Outputs

<!-- Add screenshots of model outputs from predictions.ipynb here -->

## Conclusion

This project showcases the application of deep learning for symbol spotting in architectural plans, enabling automated recognition of design elements.

---

Feel free to fill in the placeholders with your figures and screenshots as needed!
