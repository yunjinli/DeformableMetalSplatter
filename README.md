# Deformable Instance MetalSplatter
This project is forked from an amazing work [MetalSplatter](https://github.com/scier/MetalSplatter). It provides support for running deformable splats as well as instance segmented splats. 

Render deformable 3D Gaussian Splats using Metal on Apple platforms. Tested on : 
- Iphone 15, Iphone 17 Air 
- Ipad Pro (M1)
- Macbook Pro (M5)



# Installation
Please follow the steps in original [README](./README_Orig.md) to setup the project in xcode. 

# TODOS
- [x] Half precision inference for the MLP
- [x] Adding different rendering mode (depth ✅, instances ✅) -> no class support right now
- [x] Adding support to click objects
- [x] add option to not use instance mode if clusters file is not provided (toggle disabled + warning shown)
- [ ] add optional speedup via static vs dynamic splat masking (export_static_mask.py ready, Swift integration pending)
- [ ] Update BibTex after 3DV proceedings are published 

# Usage
By selecting a folder in the startup page, the app loads the ```weights.bin```,```clusters.bin``` and ```point_cloud.ply``` inside the directory. You can download an example scene [as_novel_view](https://drive.google.com/drive/folders/1s6oHkxfwywKQ4eb6WwNz9CQr80wIQqa9?usp=sharing) from NeRF-DS trained with [TRASE](https://github.com/yunjinli/TRASE). 

There is a scroll bar for adjusting the time but you can also let it play by deactivating the manual time setting. You can toggle the additional dropdown to unlock additional options such as the [TRASE](https://github.com/yunjinli/TRASE) based instance segmentation "Show Clusters" and depth based visualisation "Depth" (based on camera viewpoint). Clicking on clusters will isolate them in the visualisation, showing the whole scene again can be done via "Show all" button. 

The gestures for X/Y Panning, Orbit, Zoom in/out are also implemented. As some scenes (such as sear-stake) may have a flipped coordinate system we add some buttons to change the coordinate axis in the dropdown. 

## Using your own Scenes 

1) For dynamic splat scenes  

Output should be the base path of your input folder, where ```point_cloud.ply``` is stored.

Export the deform.pth via : 
```bash 
python export_deform_weights.py --model <path-to-deform.pth> --output <path to deform output.bin>
```

2) Optionally for clusters (store in same path as ```point_cloud.ply``` just like for 1) :

Train your scene with [TRASE](https://github.com/yunjinli/TRASE) and run the export script : 
```bash 
python export_clusters_bin.py --model <clusters.pt> --output <path to clusters output.bin>
```

# Demo

## Dynamic Splats + Instances on iPhone 17 Air 

<video src="assets/Iphone_sear_stake.mp4" controls width="600"></video>

## Dynamic Splats + Instances on iPad Pro (M1)

<video src="assets/ipad_split_cookie.mp4" controls width="600"></video>


## Dynamic Splats on iPhone 15

https://github.com/user-attachments/assets/fee3bc1f-168a-4adb-b358-5274d74e6000


# Acknowledgments
This project is a fork of MetalSplatter created by Sean Cier.

Original code is licensed under the MIT License (Copyright © 2023 Sean Cier).

Modifications and new features are licensed under MIT License (Copyright © 2026 Jim Li).

# References

If you find this useful for your own work, particularly the instance segmentation and selection please consider citing the related TRASE paper :

```
@article{li2024trase,
    title={TRASE: Tracking-free 4D Segmentation and Editing},
    author={Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal={arXiv preprint arXiv:2411.19290},
    year={2024}
}
```