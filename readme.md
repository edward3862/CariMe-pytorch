# CariMe: Unpaired Caricature Generation with Multiple Exaggerations

The official pytorch implementation of the paper "CariMe: Unpaired Caricature Generation with Multiple Exaggerations"

![examples](images/examples.png)

>CariMe: Unpaired Caricature Generation with Multiple Exaggerations
>
>Zheng Gu, Chuanqi Dong, Jing Huo, Wenbin Li, and Yang Gao
>
>Paper: https://arxiv.org/abs/2010.00246



## Prerequisites
- Python 3
- Pytorch 1.5.1
- scikit-image

## Preparing Dataset
- Get the [Webcaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm) dataset, unzip the dataset to the `data` folder and align the dataset by running the following script:
```shell script
python alignment.py
```

## Training
Train the Warper:
```shell script
python train_warper.py
```
Train the Styler:
```shell script
python train_styler.py
```

## Testing
- Test the Warper only:
```shell script
python test_warper.py --scale 1.0
```

- Test the Styler only:
```shell script
python test_styler.py 
```

- Generate caricatures with both exaggeration and style transfer:
```shell script
python main_generate.py --model_path_warper pretrained/warper.pt --model_path_styler pretrained/styler.pt
```


- Generate caricatures with both exaggeration and style transfer for a single image:
```shell script
python main_generate_single_image.py 
--model_path_warper pretrained/warper.pt \ 
--model_path_styler pretrained/styler.pt \
--input_path images/Meg Ryan/P00015.jpg \
--generate_num 5 \
--scale 1.0 
```

The above command will translate the input photo into 5 caricatures with different exaggerations and styles:

![examples](images/Meg%20Ryan/P00015_gen.jpg)


## Pretrained Models
The pre-trained models are shared [here](https://drive.google.com/drive/folders/1hBdCqWZ-kqvVLOCz-j9faLNkIbifBr3t?usp=sharing).

## Citation
If you use this code for your research, please cite our paper.

    @article{gu2020carime,
    title={CariMe: Unpaired Caricature Generation with Multiple Exaggerations},
    author={Gu, Zheng and Dong, Chuanqi and Huo, Jing and Li, Wenbin and Gao, Yang},
    journal={arXiv preprint arXiv:2010.00246},
    year={2020}
    }


## Reference
Some of our code is based on [FUNIT](https://github.com/NVlabs/FUNIT) and [UGATIT](https://github.com/znxlwm/UGATIT-pytorch).