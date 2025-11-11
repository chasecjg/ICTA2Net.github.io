import os
import requests

# import torch
import torch
import torch.nn as nn
from torch.autograd import Variable


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# class EDMLoss(nn.Module):
#     def __init__(self):
#         super(EDMLoss, self).__init__()

#     def forward(self, p_target, p_estimate):
#         assert p_target.shape == p_estimate.shape
#         # cdf for values [1, 2, ..., 10]
#         cdf_target = torch.cumsum(p_target, dim=1)
#         # cdf for values [1, 2, ..., 10]
#         cdf_estimate = torch.cumsum(p_estimate, dim=1)
#         cdf_diff = cdf_estimate - cdf_target
#         samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
#         return samplewise_emd.mean()
    
import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, score1, score2, label):
        # label 是 1 或 -1，表示图像1比图像2好或图像2比图像1好
        return self.margin_ranking_loss(score1, score2, label)


class RankNetLoss_v1(nn.Module):
    def __init__(self):
        super(RankNetLoss_v1, self).__init__()

    def forward(self, pred_scores_1, pred_scores_2, labels):
        """
        计算RankNet损失
        :param pred_scores_1: 模型对第一张图像的预测分数，形状为 (batch_size,)
        :param pred_scores_2: 模型对第二张图像的预测分数，形状为 (batch_size,)
        :param labels: 真实标签，1表示第一张图像质量更好，-1表示第二张图像质量更好，形状为 (batch_size,)
        :return: RankNet损失值
        """
        score_diffs = pred_scores_1 - pred_scores_2
        loss = torch.log(1 + torch.exp(-labels * score_diffs)).mean()
        return loss
    

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, pred_score_1, pred_score_2, labels):
        """
        计算RankNet损失
        :param pred_score_1: 模型对第一个样本的预测分数，形状为 (batch_size,)
        :param pred_score_2: 模型对第二个样本的预测分数，形状为 (batch_size,)
        :param labels: 真实标签，1表示第一个样本优于第二个样本，
                       -1表示第二个样本优于第一个样本，
                       0表示两者无差异，形状为 (batch_size,)
        :return: RankNet损失值
        """
        score_difference = pred_score_1 - pred_score_2
        # 计算概率
        p_ij = 1 / (1 + torch.exp(-score_difference))
        # 为了避免数值计算问题，对 p_ij 进行裁剪
        p_ij = torch.clamp(p_ij, min=1e-7, max=1 - 1e-7)
        # 对标签进行映射
        mapped_labels = 0.5 * (1 + labels)
        # 根据映射后的标签计算交叉熵损失
        loss = -torch.mean(mapped_labels * torch.log(p_ij) + (1 - mapped_labels) * torch.log(1 - p_ij))
        return loss
    
    

# import torch
# import torch.nn as nn


# class XT_Net_loss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(XT_Net_loss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings, labels):
#         """
#         前向传播计算监督对比损失。
#         :param embeddings: 形状为 (batch_size, embedding_dim) 的嵌入向量张量。
#         :param labels: 形状为 (batch_size,) 的标签张量，用于确定正样本。
#         :return: 计算得到的监督对比损失。
#         """
#         batch_size = embeddings.size(0)
#         similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
#         print("similarity_matrix", similarity_matrix)
#         labels_expanded = labels.unsqueeze(1) == labels.unsqueeze(0)
#         print("labels_expanded", labels_expanded)
#         positive_mask = labels_expanded.float()
#         negative_mask = 1 - positive_mask - torch.eye(batch_size, device=embeddings.device)
#         print("positive_mask", positive_mask)
#         print("negative_mask", negative_mask)
#         positive_logits = similarity_matrix * positive_mask
#         print("positive_logits", positive_logits)
#         positive_logits = positive_logits - (1 - positive_mask) * 1e9
#         positive_logits = torch.logsumexp(positive_logits, dim=1)

#         all_logits = similarity_matrix * negative_mask
#         all_logits = all_logits - (1 - negative_mask) * 1e9
#         all_logits = torch.logsumexp(all_logits, dim=1)

#         loss = -positive_logits + all_logits
#         loss = loss.mean()
#         return loss


import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Computes the contrastive loss

    Args:
        - k: the number of transformations per batch
        - temperature: temp to scale before exponential

    Shape:
        - Input: the raw, feature scores.
                tensor of size :math:`(k x minibatch, F)`, with F the number of features
                expects first axis to be ordered by transformations first (i.e., the
                first "minibatch" elements is for first transformations)
        - Output: scalar
    """

    def __init__(self, k: int, temp: float, abs: bool, reduce: str) -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.abs = abs
        self.reduce = reduce
        #         self.iter = 0

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        n_samples = len(out)
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        #         if (self.iter % 100) == 0:
        #             print(sim)
        #          self.iter += 1

        sim = torch.exp(sim * self.temp)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(self.k):
            start, end = i * (n_samples // self.k), (i + 1) * (n_samples // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss






from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        device, dtype = y_pred[0].device, y_pred[0].dtype

        target = torch.ones_like(y_true[0]).to(device).to(dtype)
        # target = target.unsqueeze(-1)
        # Set indices where y_true1 < y_true2 to -1
        target[y_true[0] < y_true[1]] = -1.0

        return F.margin_ranking_loss(
            y_pred[0],
            y_pred[1],
            target,
            margin=self.margin
        )


class RegRankLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.reg_loss = nn.MSELoss(reduction="mean")
        self.rank_loss = RankLoss(margin)

    def forward(
        self,
        y_pred: Tuple[torch.Tensor, torch.Tensor],
        y_true: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        y_true = (y_true[0].view(-1, 1), y_true[1].view(-1, 1))


        loss_reg = (
            self.reg_loss(y_pred[0], y_true[0]) +
            self.reg_loss(y_pred[1], y_true[1])
        ) / 2.0

        loss_rank = self.rank_loss(y_pred, y_true)
        loss = loss_reg + loss_rank
        return loss, loss_reg, loss_rank
    
    


# 对比损失计算函数
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    实现图像-文本对称对比损失
    """
    def __init__(self, temperature=0.07):
        """
        初始化对比损失模块
        
        参数:
            temperature: 温度缩放参数，控制相似度分布的尖锐程度
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, vis_emb, text_emb):
        """
        前向传播计算对比损失
        
        参数:
            vis_emb: [B, D] - 投影后的视觉特征
            text_emb: [B, D] - 投影后的文本特征
            
        返回:
            对称对比损失值
        """
        # 特征归一化
        vis_emb = F.normalize(vis_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(vis_emb, text_emb.t()) / self.temperature
        
        # 自动生成标签
        batch_size = vis_emb.size(0)
        labels = torch.arange(batch_size, device=vis_emb.device)
        
        # 对称损失计算
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2



# def group_contrastive_loss(vis_feats, txt_feats, group_ids, temperature=0.07):
#     """
#     严格遵循对比学习公式的正负样本划分
#     vis_feats: (proj_vis1, proj_vis2) 每组两个图像特征 [B,D] each
#     txt_feats: (proj_txt1, proj_txt2) 对应文本特征 [B,D] each
#     group_ids: [B] 组标识
#     """
#     # 合并所有特征
#     all_vis = torch.cat([vis_feats[0], vis_feats[1]], dim=0)  # [2B, D]
#     all_txt = torch.cat([txt_feats[0], txt_feats[1]], dim=0)  # [2B, D]
#     expanded_gids = group_ids.repeat(2)  # [2B]
    
#     # 归一化
#     all_vis = F.normalize(all_vis, dim=1)
#     all_txt = F.normalize(all_txt, dim=1)
    
#     # 计算相似度矩阵
#     sim_matrix = torch.mm(all_vis, all_txt.t()) / temperature  # [2B, 2B]
    
#     # 构建正样本掩码 (同组样本)
#     pos_mask = torch.eq(
#         expanded_gids.unsqueeze(1), 
#         expanded_gids.unsqueeze(0)
#     ).float() - torch.eye(2*len(group_ids)).to(group_ids.device)
#     print(pos_mask)
#     # 计算对比损失
#     exp_sim = torch.exp(sim_matrix)
#     pos = (exp_sim * pos_mask).sum(dim=1)  # 同组正样本
#     neg = exp_sim.sum(dim=1) - pos  # 不同组负样本
    
#     loss = -torch.log(pos / (pos + neg + 1e-8)).mean()
#     return loss


# import matplotlib.pyplot as plt

# class GroupContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07, debug=False):
#         super().__init__()
#         self.temperature = temperature
#         self.debug = debug
        
#     def visualize_mask(self, mask, group_ids):
#         """可视化组掩码关系"""
#         plt.figure(figsize=(10, 8))
        
#         # 创建带组标记的坐标轴
#         unique_gids = torch.unique(group_ids)
#         for gid in unique_gids:
#             idx = torch.where(group_ids == gid)[0][0]
#             plt.axvline(x=idx-0.5, color='r' if gid%2 else 'b', linestyle='--', alpha=0.3)
#             plt.axhline(y=idx-0.5, color='r' if gid%2 else 'b', linestyle='--', alpha=0.3)
        
#         # 绘制矩阵
#         plt.imshow(mask.cpu().numpy(), cmap='Blues')
#         plt.colorbar()
#         plt.title(f"Group Mask (Temperature={self.temperature})\n"
#                  f"Group IDs: {group_ids.cpu().numpy()}")
#         plt.xlabel("Sample Index")
#         plt.ylabel("Sample Index")
#         # 保存
#         plt.savefig("/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/utils/vis,pdf", dpi=300)
#         # plt.show()
    
#     def forward(self, vis_feats, txt_feats, group_ids):
#         """
#         vis_feats: (proj_vis1, proj_vis2) 每组两个图像特征 [B,D]
#         txt_feats: (proj_txt1, proj_txt2) 对应文本特征 [B,D] 
#         group_ids: [B] 组标识
#         """
#         # 合并特征
#         all_vis = torch.cat(vis_feats, dim=0)  # [2B, D]
#         all_txt = torch.cat(txt_feats, dim=0)  # [2B, D]
#         expanded_gids = group_ids.repeat(2)  # [2B]
        
#         # 归一化
#         all_vis = F.normalize(all_vis, dim=1)
#         all_txt = F.normalize(all_txt, dim=1)
        
#         # 计算相似度矩阵
#         sim_matrix = torch.mm(all_vis, all_txt.t()) / self.temperature  # [2B, 2B]
        
#         # 构建正样本掩码 (对角线置零)
#         pos_mask = torch.eq(
#             expanded_gids.unsqueeze(1), 
#             expanded_gids.unsqueeze(0)
#         ).float()
#         pos_mask.fill_diagonal_(0)  # 关键修改：排除自身对比
        
#         if self.debug:
#             print("="*50)
#             print(f"Group IDs (expanded): {expanded_gids.cpu().numpy()}")
#             print(f"Similarity Matrix:\n{sim_matrix.detach().cpu().numpy().round(2)}")
#             print(f"Positive Mask:\n{pos_mask.cpu().numpy()}")
#             self.visualize_mask(pos_mask, expanded_gids)
        
#         # 计算对比损失
#         exp_sim = torch.exp(sim_matrix)
#         pos = (exp_sim * pos_mask).sum(dim=1)  # 同组正样本
#         neg = exp_sim.sum(dim=1) - pos - exp_sim.diag()  # 排除对角线
        
#         loss = -torch.log(pos / (pos + neg + 1e-8)).mean()
#         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class GroupContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, debug: bool = False):
        super().__init__()
        self.temperature = temperature
        self.debug = debug
        
    def _validate_mask(self, mask: torch.Tensor, group_ids: torch.Tensor) -> bool:
        """验证掩码是否正确标记了同组样本"""
        gids = group_ids.cpu().numpy()
        mask = mask.cpu().numpy()
        n_samples = len(gids)
        
        # 检查对角线是否为0
        if np.any(np.diag(mask) != 0):
            print("错误：对角线元素不为0")
            return False
        
        # 检查同组样本是否被正确标记
        for i in range(n_samples):
            for j in range(n_samples):
                if gids[i] == gids[j] and i != j:
                    if mask[i,j] != 1:
                        print(f"错误：同组样本({i},{j})未被标记为正样本")
                        return False
                elif mask[i,j] == 1:
                    print(f"错误：不同组样本({i},{j})被错误标记为正样本")
                    return False
        return True
    
    def visualize_mask(self, mask: torch.Tensor, group_ids: torch.Tensor, save_path: str = None):
        """增强版掩码可视化"""
        mask_np = mask.cpu().numpy()
        gids = group_ids.cpu().numpy()
        unique_gids = np.unique(gids)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制组边界
        for gid in unique_gids:
            indices = np.where(gids == gid)[0]
            if len(indices) > 0:
                color = plt.cm.tab20(gid % 20)
                start, end = indices[0]-0.5, indices[-1]+0.5
                for pos in [start, end]:
                    plt.axvline(pos, color=color, linestyle='--', alpha=0.7)
                    plt.axhline(pos, color=color, linestyle='--', alpha=0.7)
        
        # 绘制掩码矩阵
        im = plt.imshow(mask_np, cmap='Blues', vmin=0, vmax=1)
        
        # 标记样本编号和组ID
        for i in range(len(gids)):
            for j in range(len(gids)):
                if mask_np[i,j] == 1:
                    plt.text(j, i, f"{gids[i]}", ha='center', va='center', color='red')
                elif i == j:
                    plt.text(j, i, "X", ha='center', va='center', color='black')
        
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Positive Sample Mask\nGroup IDs: {gids}")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def forward(self, 
               vis_feats: Tuple[torch.Tensor, torch.Tensor], 
               txt_feats: Tuple[torch.Tensor, torch.Tensor], 
               group_ids: torch.Tensor) -> torch.Tensor:
        """
        严格验证的组对比损失计算
        
        参数:
            vis_feats: (proj_vis1, proj_vis2) 每组两个图像特征 [B,D]
            txt_feats: (proj_txt1, proj_txt2) 对应文本特征 [B,D]
            group_ids: [B] 组标识
            
        返回:
            对比损失值
        """
        # 合并特征
        all_vis = torch.cat(vis_feats, dim=0)  # [2B, D]
        all_txt = torch.cat(txt_feats, dim=0)  # [2B, D]
        expanded_gids = group_ids.repeat(2)    # [2B]
        
        # 归一化
        all_vis = F.normalize(all_vis, dim=1)
        all_txt = F.normalize(all_txt, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(all_vis, all_txt.t()) / self.temperature  # [2B, 2B]
        
        # 构建正样本掩码
        pos_mask = torch.eq(
            expanded_gids.unsqueeze(1), 
            expanded_gids.unsqueeze(0)
        ).float()
        pos_mask.fill_diagonal_(0)  # 排除自对比
        
        # 验证掩码
        # if not self._validate_mask(pos_mask, expanded_gids):
        #     raise ValueError("正样本掩码验证失败！")
        
        # 调试信息
        if self.debug:
            print("="*50)
            print(f"Group IDs (expanded): {expanded_gids.cpu().numpy()}")
            print(f"Similarity Matrix:\n{sim_matrix.detach().cpu().numpy().round(2)}")
            print(f"Positive Mask:\n{pos_mask.cpu().numpy()}")
            self.visualize_mask(
                pos_mask, 
                expanded_gids,
                save_path="/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/utils/vis"
            )
        
        # 计算对比损失
        exp_sim = torch.exp(sim_matrix)
        pos = (exp_sim * pos_mask).sum(dim=1)  # 同组正样本
        neg = exp_sim.sum(dim=1) - pos - exp_sim.diag()  # 排除对角线
        
        loss = -torch.log(pos / (pos + neg + 1e-8)).mean()
        return loss

if __name__ == '__main__':
    # 初始化
    contrast_loss = GroupContrastiveLoss(
        temperature=0.05,
        debug=True  # 开启调试模式
    )
    proj_vis1 = torch.randn(8, 512)
    proj_vis2 = torch.randn(8, 512)
    proj_txt1 = torch.randn(8, 512)
    proj_txt2 = torch.randn(8, 512)
    batch_group_ids = torch.tensor([0, 0, 1, 1, 3, 5, 6, 6])
    # 在训练循环中
    loss = contrast_loss(
        vis_feats=(proj_vis1, proj_vis2),
        txt_feats=(proj_txt1, proj_txt2),
        group_ids=batch_group_ids
    )