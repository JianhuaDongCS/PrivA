# Accurate Personal Information is Not Necessary: User Attribute Anonymization for Recommender Systems

This is our implementation for the paper:

Fan Liu, Jianhua Dong, Zhiyong Cheng, Liqiang Nie, Mohan Kankanhalli. 

**Please cite our paper if you use our codes or datasets. Thanks!**

## environment:

Python 3.7.4,
Pytorch 1.11.0.

## Reproducibility

Train and evaluate our model:

ML-100K Age
```
python main_age.py --dataset 'ml-100k/'  --weight_decay 1e-5 --lr 0.0001 --lambdal 1.0 --lambdas 1.0 --attribute_dim 7 --c 0.001
```

ML-100K Gender
```
python main_gender.py --dataset 'ml-100k/'  --weight_decay 1e-5 --lr 0.001 --attribute_dim 2 --c 0.01
```

ML-100K Occupation
```
python main_occupation.py --dataset 'ml-100k/'  --weight_decay 1e-5 --lr 0.0001 --attribute_dim 21 --c 0.0
```

ML-1M Age
```
python main_age.py --dataset 'ml-1m/'  --weight_decay 1e-5 --lr 0.0001 --lambdal 1.0 --lambdas 0.1 --attribute_dim 7 --c 0.00001
```

ML-1M Gender
```
python main_gender.py --dataset 'ml-1m/'  --weight_decay 1e-5 --lr 0.0001 --attribute_dim 2 --c 0.001
```

ML-1M Occupation
```
python main_occupation.py --dataset 'ml-1m/'  --weight_decay 1e-5 --lr 0.0001 --attribute_dim 21 --c 0.0001
```
