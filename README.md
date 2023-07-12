# flowers-sort
 Based on Tensorflow

### 目录结构

```
flowers-sort
│
│  README.md
│  flower_sort.py
│  
├─dataset
│  ├─test
│  │   ├─daisy   
│  │   ├─iris   
│  │   ├─rose 
│  │   ├─sunflower
│  │   ├─tulips   
│  │   └─wild_rose
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
- `flower_sort.py` ：基于训练模型使用Tkinter实现的花卉种类查询工具
- `dataset/` ：六种花卉数据集
- `models/` ：存放模型
    - `flower_simple.py`，一个简单的CNN定义
    - `flower_resnet50.py`，Resnet50预训练模型微调
- `tools/` ：
    - `dataset_opt.py`，数据集处理与划分
    - `test_model.py`，模型评估
    - `train_model.py`，训练模型
---
### TODO
- test_model.py
- flower_sort.py