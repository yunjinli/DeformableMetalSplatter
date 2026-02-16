# Deformable Instance MetalSplatter
This project is forked from an amazing work [MetalSplatter](https://github.com/scier/MetalSplatter). It provides support for running deformable splats as well as instance segmented splats. 

Render deformable 3D Gaussian Splats using Metal on Apple platforms. Tested on : 
- Iphone 15, Iphone 17 Air 
- Ipad Pro (M1)
- Macbook Pro (M5)



# Installation
Please follow the steps in original [README](./README_Orig.md) to setup the project in xcode. 

# TODOS
- [ ] add optional speedup via static vs dynamic splat masking (export_static_mask.py ready, Swift integration pending)
- [ ] Update BibTex after 3DV proceedings are published
- [ ] ...

# Usage
By selecting a folder in the startup page, the app loads the ```weights.bin```, ```clusters.bin``` and ```point_cloud.ply``` inside the directory. You can download example scenes from [here](https://drive.google.com/drive/folders/1WNnabmOoLe5aX9xD_rgFajjIEYVGgpVJ?usp=sharing) trained with [TRASE](https://github.com/yunjinli/TRASE). 

There is a scroll bar for adjusting the time but you can also let it play by deactivating the manual time setting. You can toggle the additional dropdown to unlock additional options such as the [TRASE](https://github.com/yunjinli/TRASE) based instance segmentation "Show Clusters" and depth based visualisation "Depth" (based on camera viewpoint). Clicking on clusters will isolate them in the visualisation, showing the whole scene again can be done via "Show all" button. 

The gestures for X/Y Panning, Orbit, Zoom in/out are also implemented. As some scenes (such as sear-stake) may have a flipped coordinate system we add some buttons to change the coordinate axis in the dropdown. 

## Using your own Scenes 

### Step 1: Create env

```
conda create -n trase_model_converter python=3.10 -y
conda activate trase_model_converter
pip install torch "numpy<2" torchinfo
```

### Step 2: For dynamic splat scenes  

Output should be the base path of your input folder, where ```point_cloud.ply``` is stored.

Export the deform.pth via : 
```bash 
python export_deform_weights.py --model <path-to-deform.pth> --output <path to deform output.bin>
```

### Step 3 TRASE clustering: Optionally for clusters (store in same path as ```point_cloud.ply``` just like for 1) :

Train your scene with [TRASE](https://github.com/yunjinli/TRASE) and run the export script : 
```bash 
python export_clusters_bin.py --model <clusters.pt> --output <path to clusters output.bin>
```

### Step 4 MobileCLIP querying: Optionally for cluster scenes (perform step 3 first) 

Download the Mobileclip coreml models from [our google drive link](https://drive.google.com/drive/folders/1pilHtEPD7ShOhJpYHbJUgB3HiBgB3CZg?usp=sharing) and put it in ```DeformableMetalSplatter/``` (whole path is then ```DeformableMetalSplatter/coreml_models/```)

You can then move to a certain view and run the ```Encode Clusters CLIP``` button to encode the features, after which the search bar can be used to query clusters with text. ```topk``` setting can be used to decide which amount of clusters can be queried.  

# Demo

## Dynamic Splats + Instances on iPhone 17 Air 

https://github.com/user-attachments/assets/eca6ebd1-7a0b-4ce4-836a-c89b5228afbe


## Dynamic Splats + Instances on iPad Pro (M1)


https://github.com/user-attachments/assets/4f7c2aa8-b9bd-4563-ad21-1f4d2a68379c

## MobileCLIP querying on iPhone 17 Air 


## Dynamic Splats + Instances on iPhone 15

https://github.com/user-attachments/assets/3cc26612-be2c-402c-98aa-e538ddf6d732

## MobileCLIP Encoding and Instance Querying on iPhone 17 Air

https://github.com/user-attachments/assets/c78bf20e-2da5-473e-947b-b0c7bbb5d851

## MobileCLIP Encoding and Instance Querying on Macbook Pro M5


https://github.com/user-attachments/assets/789c4074-9b70-4b5d-b072-4bc0e3782901


https://github.com/user-attachments/assets/d3423e47-947e-4d1c-a842-eedab2826cf9


# Acknowledgments
This project is a fork of MetalSplatter created by Sean Cier.

Original code is licensed under the MIT License (Copyright © 2023 Sean Cier).

Modifications and new features are licensed under MIT License (Copyright © 2026 Jim Li).

# References

If you find this useful for your own work, particularly the 3D segmentation please consider taking a look at TRASE's [code](https://github.com/yunjinli/TRASE), [paper](https://arxiv.org/pdf/2411.19290), and [website](https://yunjinli.github.io/project-sadg/). Give us a star if you find it interesting :)

```
@article{li2024trase,
    title={TRASE: Tracking-free 4D Segmentation and Editing},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal={arXiv preprint arXiv:2411.19290},
    year={2024}
}
```

## Contributors

<div align="center">
  <a href="https://github.com/yunjinli"><img src="https://images.weserv.nl/?url=github.com/yunjinli.png&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px" title="Jim" /></a>
  &nbsp; &nbsp; 
  <a href="https://github.com/Cedric-Perauer"><img src="https://images.weserv.nl/?url=github.com/Cedric-Perauer.png&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px" title="Cedric" /></a>
</div>
