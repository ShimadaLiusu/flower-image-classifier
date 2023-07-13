# flowers-sort
 Based on Tensorflow

## How to use

### 一键安装依赖
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

## 目录结构

```
flowers-sort
│
│  README.md
│  flower_sort.py
│  
├─dataset
│  │
│  ├─train
│  │   ├─daisy   
│  │   ├─iris   
│  │   ├─rose 
│  │   ├─sunflower
│  │   ├─tulips   
│  │   └─wild_rose
│  │          
│  └─val
│      ├─daisy   
│      ├─iris   
│      ├─rose 
│      ├─sunflower
│      ├─tulips   
│      └─wild_rose
│      
├─models
│  │  flower_alexnet.py
│  │  flower_resnet50.py
│  │  flower_simple.py
│  │  
│  └─__pycache__
│          
└─tools
        dataset_opt.py
        test_model.py
        train_model.py
```
`flower_sort.py` ：使用Tkinter实现的花卉种类查询工具，在`./checkpoint/`放训练好的模型权重后使用  


`dataset/` ：六种花卉数据集   

`models/` ：存放模型
- `flower_simple.py`，一个简单的CNN
- `flower_alexnet.py`，手搓AlexNet
- `flower_resnet50.py`，Resnet50预训练模型微调   

`tools/` ：
- `dataset_opt.py`，数据集处理与划分
- `train_model.py`，训练模型，训练好的模型存放在`./checkpoint/`



## TODO
- requirements.txt
- 闲了写写其他网络跑一下训练
- 闲了写`test_model.py`