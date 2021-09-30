This is a pytorch implementation of the paper

>  Junjun Pan, Siyuan Wang, Junxuan Bai,  Ju Dai. Diverse Dance Synthesis via Keyframes with Transformer Controllers. Computer Graphics Forum, 2021.

In this paper, we propose a novel keyframe-based motion generation network based on multiple constraints, which can achieve diverse dance synthesis via learned knowledge.

## Dependencies

- Python 3.6
- Pytorch 1.7.0
- Matplotlib
- NumPy
- Pyyaml
- ...

(1) Define skeleton information in ./global_info; (2) Put motion data(.bvh) in ./data; (3) Modify training or test parameters in ./config.py.

We use contemporary data from paper 

> Andreas Aristidou, Qiong Zeng, Efstathios Stavrakis, KangKang Yin, Daniel Cohen-Or, Yiorgos Chrysanthou, and Baoquan Chen. 2017. Emotion control of unstructured dance movements. In Proceedings of the ACM SIGGRAPH / Eurographics Symposium on Computer Animation (SCA '17).

## For data preparation

```
python generate_data.py --dataset Cyprus --downsample 1
```

## For training

```
python train.py --train prediction --data_path data/Cyprus_out/
```

## For testing

```
python test.py --test prediction --data_path data/Cyprus_out/
```



The paper and demo will be published soon.



