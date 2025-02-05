## 1.创建setup.py文件
from setuptools import setup,find_packages  # 改为从 setuptools 导入

setup(
    name="hm_message",
    version="1.0",
    description="模块描述xxx",
    long_description="长描述xx",
    author="wm",
    packages=find_packages(),
)


## 2.构建模块      python3 setup.py build
## 3.生成发布压缩包 python3 setup.py sdist
########=================================
##4.安装压缩包
#     pip3 install hm_message-1.0.tar.gz
