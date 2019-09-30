# ZO-minmax


Prerequisites
-----------------------

The code is tested with python3.7 and TensorFlow v1.13. Please use miniConda to manage your Python environments.
The following packages are required:

```
tensorflow-gpu>=1.13.0
scipy=1.1.0
```


Black-box ensemble evasion attack (adversarial example)
--------------------------------------------------

To download the pre-trained models:

```
python3 setup_imagenet.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagesnetdata`. 


ZO-minmax attack:

run 
```
python3 Main_minmax_universal.py --minmax=1
```
You can change methods or any hyperparameter in args.

ZO-Finite-Sum case:

run 
```
python3 Main_minmax_universal.py --minmax=0
```

Black-box  poisoning  attack  against  logistic  regression  model
---------------------------------------------------


run 
```
python3 Main_poison_attack.py
```

Compare stable point


run 
```
python3 Main_comparison_STABLEOPT
```
