# Deep Spiking Neural Networks for Large Vocabulary Automatic Speech Recognition
This repository provides the code to demonstrate and facillitate the implementation of SNN-based ASR systems with Pytorch-Kaldi toolbox. The description of the system and implementation can be found at our arxiv paper [click here](https://arxiv.org/abs/1911.08373).

## Requirements/Prerequisites
In this project, the acoustic modelling is handled by the feedforward spiking neural networks that implemented in **Pytorch (version 1.0.1)** with customized Linear and dorpout layers etc. While feature extraction, label computation, and decoding are performed with the **Kaldi toolkit**. To ensure the code is able to run properly, please follow the instrcutions provided [here](https://github.com/mravanelli/pytorch-kaldi) to install the Kaldi and Pytorch-Kaldi toolbox.

## How to use the code?
1. Add/Repalce the **functional.py**, **snn.py**, **neural_networks.py** files under the home directory of **Pytorch-Kaldi** installation.
2. Add files inside the **cfg** and **proto** folders into the respective directories of the **Pytorch-Kaldi** installation.
3. Set the configuration files (location of the feature, labels, out_folder etc.) in step 2 properly according to the instructions provided at ["Description of the configuration files"](https://github.com/mravanelli/pytorch-kaldi) section. 

## Contact
For queries please contact jibin.wu@u.nus.edu
