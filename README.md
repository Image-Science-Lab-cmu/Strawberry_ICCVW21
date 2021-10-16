# Fine-Grain Prediction of Strawberry Freshness using Subsurface Scattering

## Dataset
Download the dataset [here](https://cmu.box.com/s/k8k1fz5i5x85hjj4g4oo6yaouo9oqu77).
The post-processed captures are in processed/.

## Usage
### Preprocessing
[process_save_all.py](preprocessing/process_save_all.py) loads raw captures and generates the post-processed images described in the paper.

### Training
[strawberry.py](training/strawberry.py) defines a single strawberry object containing the various structured illumination captures.
The Strawberry class provides an interface to obtain image stacks for each scattering feature.

## Citation
```
@inproceedings{klotz2021fine,
title = {Fine-Grain Prediction of Strawberry Freshness using Subsurface Scattering},
author = {Klotz, Jeremy and Rengarajan, Vijay and Sankaranarayanan, Aswin C.},
booktitle = {ICCV Workshop on Large-Scale Fine-Grained Food AnalysIs (LargeFineFoodAI)},
year = {2021},
}
```
