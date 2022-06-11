# MCENet
This is a PyTorch implementation of the model in this paper:

[Multi-source collaborative enhanced for remote sensing images semantic segmentation](https://doi.org/10.1016/j.neucom.2022.04.045)

## Dependencies
* PyTorch 1.6.0
* torchvision 0.7.0
* torchsummary 
* numpy
* apex
* imageio
* tqdm
* opencv-python
* Pillow
* tensorboard
* tifffile

## Tips
Please modify the ```dataloader/dataset.py``` according to the name of the images in the dataset

Train:
```
python main.py --savedir=' Model save path ' --lr=1e-4 --step_loss=50
```

## Citation
Please cite this paper if you use this code in your own work:
```
@inproceedings{ZhaoMCE,
  title={Multi-source collaborative enhanced for remote sensing images semantic segmentation},
  author={Jiaqi Zhao, Di Zhang, Boyu Shi, Yong Zhou, Jingyang Chen, Rui Yao, Yong Xue},
  booktitle={Neurocomputing},
  year={2022}
}
```