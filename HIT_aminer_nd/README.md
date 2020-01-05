# HIT_aminer_nd

## data_analysis.ipynb

分析数据，可视化数

## pair_dataset.ipynb

准备训练数据，将数据准备成二分类的训练数据，比如

| 文献1 | 文献2 | 标签 |
| ----- | ----- | ---- |
| id1   | id2   | 1    |

其中标签1表示两篇文献属于同一个作者，标签2表示不属于同一个作者

## preprocess.ipynb

输入训练数据，通过pipeline构建处理特征，同时训练分类器

## preprocess.py

这是.py版本，方便在各种环境下的调试。

## utils.py

工具类，获取特征