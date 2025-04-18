import os, re
from lightning.pytorch.callbacks import Callback # 用来实现一些回调自定义逻辑的

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    # 获取目录下所有的检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    if not checkpoint_files:
        return None  # 如果没有检查点文件，返回 None

    # 用正则表达式提取文件名中的 epoch 和 step
    checkpoint_files.sort(key=lambda x: (
        int(re.search(r"epoch=(\d+)_", x).group(1)),  # 提取 epoch
        int(re.search(r"step=(\d+)", x).group(1))    # 提取 step
    ), reverse=True)  # 按照 epoch 和 step 降序排列

    # 返回最新的检查点
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def keep_last_n_checkpoints(checkpoint_dir, n=2):
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')],
        key=lambda x: (
            int(x.split('_')[0].replace('epoch=', '')),  # 提取 epoch
            int(x.split('_')[1].replace('step=', '').replace('.ckpt', ''))  # 提取 step
        ),
        reverse=True  # 按照 epoch 和 step 降序排列
    )
    # 如果超过两个检查点，删除较早的
    if len(checkpoint_files) > n:
        for file_to_remove in checkpoint_files[n:]:
            file_path = os.path.join(checkpoint_dir, file_to_remove)
            os.remove(file_path)

class CheckpointCleanupCallback(Callback):
    def __init__(self, checkpoint_dir, ckpt_save_num=2):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_save_num = ckpt_save_num

    # 钩子函数
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存检查点时进行清理
        # 根据观察, 是在保存模型开始的一瞬间就触发了, 因此模型还没保存好就会触发, 因此 ckpt_save_num 这里我手动减了一个 1
        assert self.ckpt_save_num >= 1, "ckpt 检查点保存数量必须 >= 1"
        keep_last_n_checkpoints(self.checkpoint_dir, self.ckpt_save_num-1)