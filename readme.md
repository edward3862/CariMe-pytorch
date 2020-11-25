# Pytorch Code for CariMe


<p align="center">
     <img src=examples.png width=90% /> <br>
</p>

>CariMe: Unpaired Caricature Generation with Multiple Exaggerations
>
>Zheng Gu, Chuanqi Dong, Jing Huo, Wenbin Li, and Yang Gao
>
>Paper: https://arxiv.org/abs/2010.00246
>
>Abstract: Caricature generation aims to translate real photos into caricatures with artistic styles and shape exaggerations while maintaining the identity of the subject. Different from the generic image-to-image translation, drawing a caricature automatically is a more challenging task due to the existence of various spacial deformations. Previous caricature generation methods are obsessed with predicting deﬁnite image warping from a given photo while ignoring the intrinsic representation and distribution for exaggerations in caricatures. This limits their ability on diverse exaggeration generation. In this paper, we generalize the caricature generation problem from instance-level warping prediction to distribution-level deformation modeling. Based on this assumption, we present the ﬁrst exploration for unpaired CARIcature generation with Multiple Exaggerations (CariMe). Technically, we propose a Multi-exaggeration Warper network to learn the distribution-level mapping from photo to facial exaggerations. This makes it possible to generate diverse and reasonable exaggerations from randomly sampled warp codes given one input photo. To better represent the facial exaggeration and produce ﬁne-grained warping, a deformation-ﬁeld-based warping method is also proposed, which helps us to capture more detailed exaggerations than other point-based warping methods. Experiments and two perceptual studies prove the superiority of our method comparing with other state-of-the-art methods, showing the improvement of our work on caricature generation.


## Prerequisites
- Python 3
- Pytorch 1.4+
- scikit-image
- tensorboardX

## Preparing Dataset
- Get the Webcaricature dataset([link](https://cs.nju.edu.cn/rl/WebCaricature.htm)), and unzip the dataset to the `data` folder.

- Align the dataset by running:
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
Test the Warper:
```shell script
python test_warper.py --scale 1.0
```

Test the Styler:
```shell script
python test_styler.py 
```

Generate caricatures with both exaggeration and style transfer:
```shell script
python generate.py \
--model_path_warper path/to/warper/model \ 
--model_path_styler path/to/styler/model \
--generate_num K \
--scale 1.0 
```

## Citation
If you use this code for your research, please cite our paper.

    @article{gu2020carime,
    title={CariMe: Unpaired Caricature Generation with Multiple Exaggerations},
    author={Gu, Zheng and Dong, Chuanqi and Huo, Jing and Li, Wenbin and Gao, Yang},
    journal={arXiv preprint arXiv:2010.00246},
    year={2020}
    }
    