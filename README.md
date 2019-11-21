# fast-object-detection-nano
# 程序在nano上面的安装教程
## 1. 首先需要在nano上面配置pytorch
https://blog.csdn.net/donkey_1993/article/details/102794617
## 2. 然后需要编译pytorch的torch2trt使用tensorrt加速
https://github.com/NVIDIA-AI-IOT/torch2trt
## 3. 下载本工程，然后运行make.sh编译工程
sudo bash make.sh
## 4. 修改demo.py里面的视频路径。

# 训练代码
## 直接使用的是voc2007的数据集。
sudo python3 train_RFB.py

# 感谢下面两位作者
https://github.com/ruinmessi/RFBNet
https://github.com/songwsx/RFB-Person
