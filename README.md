重庆大学深度学习课程期末项目——基于深度卷积网络的肝脏分割

使用 lightning 框架 + yaml 配置文件 + wandb/tensorboard 日志记录 + pytorch 模块代码实现。

使用方法：根据自身需求修改 config.yaml, 然后 python train.py 即可。

config.yaml: baseline
configs:
    1.yaml: baseline + augment
    2.yaml: baseline + epoch 翻倍
    3.yaml: baseline + num_layers 翻倍