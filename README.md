# Point Cloud Registration and Collaborative Vehicles
 
## Abstract 
This repositoy is part of my thesis and proposes a way to register point clouds in a scene with some autonomous vehicles using an efficient and effective way of representing the environment in which each autonomous vehicle is located. More specifically, each autonomous vehicle constructs a graph whose vertices are some features of its environment that were produced after semantic segmentation of the scene it is in. These features may consist of other vehicles, pedestrians, or even fixed objects such as light poles or road signs. Finally, the last stage follows which is the matching of the graphs of the vehicles involved in a scene. Through this matching, the relative position of one autonomous vehicle in relation to another is derived, and thus a new scene can be created containing information from both vehicles filling in any gaps.

## Report
The complete thesis in Greek can be found in this [url]() or in this local [file](./docs/Thesis.pdf)

## Demo file and Weights
The weights for all segmentation networks are located [here](https://drive.google.com/drive/folders/1WgPgGwLYAaqxpuND6rLDiOaC0ZuKREts?usp=drive_link)

Demo file in Google Colab (suggested method to run the demo file): 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Sk9uEQcouigFRKC9RYAlrL7oETHiOG6?usp=drive_link)

## Install Locally
0. Create an virtual environment
```console
conda create -n myenv 
```
1. Install the [MMSEG](https://github.com/open-mmlab/mmsegmentation) package followed the instructions [here](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) (follow the **"case b"**)
2. Download the [Dataset](https://ai4ce.github.io/V2X-Sim/), all the examples used the "V2XSim-mini" dataset
3. Clone the current directory
4. Install all the necessary packages from 
[here](./requirements.txt)
```console
pip install -r /path/to/requirements.txt
```
5. Run the [demo](./demo.ipynb) file
