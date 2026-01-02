# Deformable MetalSplatter
The is a project forked from an amazing work [MetalSplatter](https://github.com/scier/MetalSplatter).

Render deformable 3D Gaussian Splats using Metal on Apple platforms (currently only tested on IOS on my iPhone 15)

# Installation
Please follow the steps in original [README](./README_Orig.md) to setup the project in xcode. 

# TODOS
- [ ] Half precision inference for the MLP
- [ ] Adding different rendering mode (depth, semantics, classes)
- [ ] Adding support to click objects
- [ ] ... 

# Usage
By selecting a folder in the startup page, the app loads the ```weights.bin``` and ```point_cloud.ply``` inside the directory. You can download an example scene [as_novel_view](https://drive.google.com/drive/folders/1s6oHkxfwywKQ4eb6WwNz9CQr80wIQqa9?usp=sharing) from NeRF-DS trained with [TRASE](https://github.com/yunjinli/TRASE). 

There is a scroll bar for adjusting the time but you can also let it play by deactivating the manual time setting.

The gestures for X/Y Panning, Orbit, Zoom in/out are also implemented. 

# Demo
https://github.com/user-attachments/assets/fee3bc1f-168a-4adb-b358-5274d74e6000

# Acknowledgments
This project is a fork of MetalSplatter created by Sean Cier.

Original code is licensed under the MIT License (Copyright © 2023 Sean Cier).

Modifications and new features are licensed under MIT License (Copyright © 2026 Jim Li).
