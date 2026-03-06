import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotterx(name, y_true, y_pred, ascore, labels):
    if 'TranAD' in name or 'DITAU-Net' in name:
        y_true = torch.roll(y_true, 1, 0)

    save_dir = os.path.join('plot', name)
    os.makedirs(save_dir, exist_ok=True)

    for dim in range(y_true.shape[1]):
        y_t = y_true[:, dim]
        y_p = y_pred[:, dim]
        l = labels[:, dim]
        a_s = ascore[:, dim]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 2.5))

        # Top plot: signal with anomalies
        ax1.plot(smooth(y_t), linewidth=0.4, color='black', label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.4, color='red', label='Predicted')
        ax3 = ax1.twinx()
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3, label='True Anomaly')

        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.set_yticks([])

        # Bottom plot: anomaly score
        ax2.plot(smooth(a_s), linewidth=0.3, color='green', label='Score')
        ax4 = ax2.twinx()
        ax4.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.3, label='Predicted Anomaly')

        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_yticks([])

        # Only for the first dimension: show legend
        if dim == 0:
            legend_elements = [
                Line2D([0], [0], color='black', lw=1, label='True'),
                Line2D([0], [0], color='red', lw=1, label='Predicted'),
                Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='True Anomaly'),
                Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Predicted Anomaly')
            ]
            fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize=8)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'dim{dim}.svg'), dpi=600, bbox_inches='tight')
        plt.close()


def plotter(name, y_true, y_pred, ascore, labels):
    if 'TranAD' in name or 'DITAU-Net' in name:
        y_true = torch.roll(y_true, 1, 0)

    save_dir = os.path.join('plot', name)
    os.makedirs(save_dir, exist_ok=True)

    # 每两个维度一组进行遍历
    for dim_group in range(0, y_true.shape[1], 2):
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 5.5))
        plt.subplots_adjust(hspace=0.3)

        # 确保dim_group+1不超出维度范围
        dims = [dim_group, dim_group + 1] if dim_group + 1 < y_true.shape[1] else [dim_group]

        # 处理每个维度
        for i, dim in enumerate(dims):
            y_t = y_true[:, dim]
            y_p = y_pred[:, dim]
            l = labels[:, dim]
            a_s = ascore[:, dim]

            # 上部分: 信号与异常区域 - 第一个维度在0,1位置，第二个维度在2,3位置
            ax1 = axes[i * 2]
            ax1.plot(smooth(y_t), linewidth=0.8, color='black', label='True')
            ax1.plot(smooth(y_p), '-', alpha=0.8, linewidth=0.8, color='red', label='Predicted')
            ax2 = ax1.twinx()
            ax2.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.25, label='True Anomaly')

            # 设置标题
            ax1.set_title(f'Dimension = {dim}')
            ax1.set_yticks([])
            if i == len(dims) - 1:  # 只有最后一个子图显示x轴标签
                ax1.set_xlabel('Timestamp')

            # 下部分: 异常分数 - 在ax1的下方
            ax3 = axes[i * 2 + 1]
            ax3.plot(smooth(a_s), linewidth=0.6, color='green', label='Anomaly Score')
            ax4 = ax3.twinx()
            ax4.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.25, label='Predicted Anomaly')

            ax3.set_ylabel('Score')
            ax3.set_yticks([])
            if i == len(dims) - 1:  # 只有最后一个子图显示x轴标签
                ax3.set_xlabel('Timestamp')

        # 添加图例（仅在图组的第一幅图上显示）
        if dim_group == 0:
            legend_elements = [
                Line2D([0], [0], color='black', lw=1, label='True'),
                Line2D([0], [0], color='red', lw=1, label='Predicted'),
                Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='True Anomaly'),
                Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Predicted Anomaly'),
                Line2D([0], [0], color='green', lw=1, label='Anomaly Score')
            ]
            fig.legend(handles=legend_elements, loc='upper center',
                       ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=8)

        plt.tight_layout()

        # 处理维度分组命名
        if len(dims) == 2:
            filename = f'dim{dims[0]}_{dims[1]}.svg'
        else:
            filename = f'dim{dims[0]}.svg'

        fig.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight')
        plt.close()
