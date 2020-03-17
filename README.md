# PPON
Pytorch implemention of "Progressive Perception-Oriented Network for Single Image Super-Resolution"

[[arXiv]](https://arxiv.org/abs/1907.10399)

<p align="center">
    <img src="figures/Structure.png" width="480"> <br />
    <em> The schematics of the Progressive Perception-Oriented Network</em>
</p>

<p align="center">
  <img height="400" src="figures/269.gif">
</p>

<p align="center">
    <img src="figures/show.jpg" width="960"> <br />
    <em> The example results</em>
</p>

<p align="center">
    <img src="figures/visualization2.png" width="960"> <br />
    <em> Visualization of intermediate feature maps</em>
</p>

<p align="center">
    <img src="figures/structure_inference.png" width="480"> <br />
    <em> The inference architecture of our PPON.</em>
</p>

<table>
   <tr>
       <td><img src="figures/278.gif"></td>
       <td><img src="figures/300.gif"></td>
   </tr>
</table>


## Testing
Pytorch 1.1
* Download [PIRM_dataset](https://drive.google.com/open?id=1T-B-EAZVeXfbUu6Frnpa6X2u7kxxRyvt) and unzip it into folder `Test_Datasets`
* Download [Checkpoint](https://drive.google.com/open?id=112qwpKHNt5TC3xyzGxYAhYaPHQqo_VrA) and put them into folder `ckpt`
* Run testing:
```bash
python test_PPON.py --test_hr_folder Test_Datasets/PIRM_Val/ --test_lr_folder Test_Datasets/PIRM_Val_LR/
```

## Training
* Download DF2K (DIV2K + Flickr2k) training datasets and rename the images to `00xxxx.png` (e.g., 003450.png)
* Convert png file to npy file
```bash
python scripts/png2npy.py --pathFrom /path/to/DF2K/ --pathTo /path/to/DF2K_decoded/
```
* Run training x4 model (stage 1, content)
```bash
python train.py --nEpochs 1000 --test_every 690 --which_model "content" --lr_steps [200, 400, 600, 800] --save_path 'ckpt_stage1'
```
(stage 2, structure)
```bash
python train.py --nEpochs 300 --test_every 138 --which_model "structure" --pixel_weight 0 --structure_weight 1 --lr_steps [100, 150, 200, 250] --save_path 'ckpt_stage2'
```
(stage3, perceptual)
```bash
python train.py --nEpochs 300 --test_every 138 --which_model "perceptual" --pixel_weight 0 --feature_weight 1 --gan_weight 0.005 --lr_steps [100, 150, 200, 250] --save_path 'ckpt_stage3'
```


## PI VS LPIPS
![PI-VS-LPIPS](figures/PI%20VS%20LPIPS.png)
As illustrated in the above picture, we can obviously see that the PI score of EPSR3 (2.2666) is even better than HR (2.3885), but EPSR3 shows unnatural and lacks proper texture and structure.

## Example Results
![Perceptual-results-1](figures/example.png)
![Perceptual-results-2](figures/example2.png)

## Citation

If you find PPON useful in your research, please consider citing:

```
@article{Hui-PPON-2019,
  title={Progressive Perception-Oriented Network for Single Image Super-Resolution},
  author={Hui, Zheng and Li, Jie and Gao, Xinbo and Wang, Xiumei},
  journal={arXiv:1907.10399v1},
  year={2019}
}
```
