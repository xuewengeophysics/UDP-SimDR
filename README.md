# UDP-SimDR

Using UDP to replace the data process method that adopted by TokenPose.

## Acknowledgements

Great thanks for these papers and their open-source codes:

+ [UDP-Pose: Unbiased Data Processing for for Human Pose Estimation](https://github.com/HuangJunJie2017/UDP-Pose)
+ [SimDR: Is 2D Heatmap Representation Even Necessary for Human Pose Estimation?](https://github.com/leeyegy/SimDR)



## Model Zoo

### Results on COCO val2017 with gt_bbox

|          Model           | Input size | AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR .5 | AR .75 | AR (M) | AR (L) |
| :----------------------: | :--------: | ----- | ----- | :----: | :----: | :----: | :---: | :---: | :----: | :----: | :----: |
| HRNet-W48 (UDP + SImDR*) |  256x192   | 0.780 | 0.935 | 0.846  | 0.753  | 0.825  | 0.808 | 0.943 | 0.865  | 0.777  | 0.855  |

### Results on COCO val2017 with Person detector has person AP of 56.4 on COCO val2017 dataset

|          Model           | Input size | AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR .5 | AR .75 | AR (M) | AR (L) |
| :----------------------: | :--------: | ----- | ----- | :----: | :----: | :----: | :---: | :---: | :----: | :----: | :----: |
| HRNet-W48 (UDP + SImDR*) |  256x192   | 0.761 | 0.904 | 0.829  | 0.727  | 0.827  | 0.812 | 0.940 | 0.871  | 0.771  | 0.871  |