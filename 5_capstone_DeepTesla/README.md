# 机器学习纳米学位

## 毕业项目: DeepTesla

## 开发环境要求

python 3.5 + Keras 1.2.1 + Tensorflow 0.12.1 + Jupyter Notebook + opencv3（具体见./environment-gpu.yml文件）

本地GPU: GTX1060 6G

操作系统：Win10 ，内存16G， CPU:i7



## 项目提交文件说明

* 项目工程文件见**'./P5-MDND-Deep-Original.ipynb'**文件，并生成了相应的**'./P5-MDND-Deep-Original.html'**文件。
* 项目报告见**'./report.pdf'**，**'./report.md'**为原markdown文件，用pandoc生成pdf。
* 文件夹中所有**‘.py’**文件均为项目辅助代码。在项目工程中有使用说明。
* 最终结果视频见**'./output/epoch10_human_machine.mp4'**



## 各文件夹说明

* **./epochs**文件夹中存放初始视频文件和CSV数据表格。为节约提交代码大小，并没有放在提交文件中，可在[百度云](https://pan.baidu.com/s/1c2J2IFA)下载。（若需要运行整个项目工程文件，只需下载数据放入该文件夹中）
* **./images**文件夹存放项目中生成的图片。
* **./models**文件夹存放项目中生成的所有模型文件。
* **./outputs**存放选做项目**vae+gan**生成的模型文件，并没有上传，可运行项目工程代码生成。
* **./test_images**文件夹存放测试视频转成的所有图片，没有上传，可运行项目工程代码生成。
* **./train_images**文件夹存放训练视频转成的所有图片，没有上传，可运行项目工程代码生成。
* **./v3**文件夹存放选做项目**cnn+rnn seq2seq**生成的文件，没有上传，可运行项目工程代码生成。
* **./vg_images**文件夹存放选做项目**vae+gan**生成的图片，没有上传，可运行项目工程代码生成。



## 项目花费时间

* 如需从头运行所以项目工程代码，大概需要5-10小时（取决于GPU运算能力）。
* 整个项目所花费时间（代码书写，报告书写），总共耗时一个半月。