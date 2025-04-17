import os, re
from lightning.pytorch.callbacks import Callback # 用来实现一些回调自定义逻辑的

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    # 获取目录下所有的检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

    if not checkpoint_files:
        return None  # 如果没有检查点文件，返回 None

    # 用正则表达式提取文件名中的 epoch 和 step
    checkpoint_files.sort(key=lambda x: (
        int(re.search(r"epoch-(\d+)_", x).group(1)),  # 提取 epoch
        int(re.search(r"step-(\d+)", x).group(1))    # 提取 step
    ), reverse=True)  # 按照 epoch 和 step 降序排列

    # 返回最新的检查点
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def keep_last_n_checkpoints(checkpoint_dir, n=2):
    checkpoint_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')],
        key=lambda x: (
            int(x.split('_')[1].replace('epoch-', '')),  # 提取 epoch
            int(x.split('_')[2].replace('step-', ''))   # 提取 step
        ),
        reverse=True  # 降序排序，最新的排在前面
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

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存检查点后进行清理
        keep_last_n_checkpoints(self.checkpoint_dir, self.ckpt_save_num)