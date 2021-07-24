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
| HRNet-W48 (UDP + SImDR*) |  256x192   | 0.779 | 0.935 | 0.845  | 0.753  | 0.823  | 0.808 | 0.946 | 0.863  | 0.776  | 0.857  |

### Results on COCO val2017 with Person detector has person AP of 56.4 on COCO val2017 dataset

|          Model           | Input size | AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR .5 | AR .75 | AR (M) | AR (L) |
| :----------------------: | :--------: | ----- | ----- | :----: | :----: | :----: | :---: | :---: | :----: | :----: | :----: |
| HRNet-W48 (UDP + SImDR*) |  256x192   | 0.759 | 0.907 | 0.824  | 0.723  | 0.826  | 0.810 | 0.943 | 0.869  | 0.768  | 0.872  |