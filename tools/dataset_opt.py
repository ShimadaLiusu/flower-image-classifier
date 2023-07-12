#原始数据中部分图片会读取错误，这里先全部读一遍，删掉有问题的（删了3张wild rose的）
import os
import shutil
import warnings
import io
from PIL import Image
import sys
sys.path.append('.')
warnings.filterwarnings("error", category=UserWarning)
base_dir = "./dataset"#删除图片的根目录
i = 0
def is_read_successfully(file):
    try:
        imgFile = Image.open(file)#这个就是一个简单的打开成功与否
        return True
    except Exception:
        return False
for parent, dirs, files in os.walk(base_dir):#(root,dirs,files)
    for file in files:
        if not is_read_successfully(os.path.join(parent, file)):
            print(os.path.join(parent, file))
            #os.remove(os.path.join(parent, file)) #移除有问题的图片
            i = i + 1
print(i)

import splitfolders
# 数据集划分
# train:validation:test=6:2:2
splitfolders.ratio(input='./dataset', output='output', seed=1337, ratio=(0.6, 0.2, 0.2))
