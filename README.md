# Is Texture Predictive for Age and Sex in Brain MRI?
This repository contains the code for the paper *Is Texture Predictive for Age and Sex in Brain MRI?*
([OpenReview](https://openreview.net/forum?id=BJxgXfab94), [arXiv](https://arxiv.org/abs/1907.10961)).

We presented this as a [poster](https://github.com/pawni/MedicalBagNet/blob/master/poster.pdf) at MIDL 2019.

## Abstract
Deep learning builds the foundation for many medical image analysis tasks where neural networks are often designed to have a large receptive field to incorporate long spatial dependencies. Recent work has shown that large receptive fields are not always necessary for computer vision tasks on natural images. We explore whether this translates to certain medical imaging tasks such as age and sex prediction from a T1-weighted brain MRI scans.

## Prerequisites
Following libraries were used for development:
```
pip install numpy pandas SimpleITK tensorboardX torch tqdm
```

## Structure
`data` contains the code for the datasets: We only used [CamCAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/) for the paper but also implemented a reader for the [IXI dataset](https://brain-development.org/ixi-dataset/). For IXI we used the [script](https://github.com/DLTK/DLTK/blob/master/data/IXI_HH/download_IXI_HH.py) provided with [DLTK](https://github.com/DLTK/DLTK) for download. `camcan_splits` contains the splits we used.

`bagnets.py` contains the network implementations adapted from [here](https://github.com/wielandbrendel/bag-of-local-features-models).

`train.py` is the actual training script.

`deploy.py` runs the evaluation and can also output the localised prediction maps.

## Usage
To run the training script, download CamCAN and change the base path in `data/camcan.py` and `train.p`. You can then run training with
```
python train.py -c <cuda_device> -l <path_to_logdirectory> --rf 9 --l2 1e-4 --attribute sex -b 1 --delayed_step 16 --scale_factor -1 --data_type camcan --opt adam
```
and run evaluation with `deploy.py`:
```
python deploy.py -m <path_to_logdirectory> -d camcan --scale 1mm --scale_factor -1 --localised --attribute age --save_path <path_to_save_predictions>
```


## Contact
For discussion, suggestions or questions don't hesitate to
contact n.pawlowski16@imperial.ac.uk .

## Citation
If you want to refer to the paper please cite:
```
@inproceedings{pawlowski:MIDLAbstract2019a,
title={Is Texture Predictive for Age and Sex in Brain {\{}MRI{\}}?},
author={Nick Pawlowski and Ben Glocker},
booktitle={International Conference on Medical Imaging with Deep Learning -- Extended Abstract Track},
address={London, United Kingdom},
year={2019},
url={https://arxiv.org/abs/1907.10961},
}
```
