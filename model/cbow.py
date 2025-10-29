import torch
import torch.nn as nn


# 连续词袋模型
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOWModel, self).__init__()
        # 上下文词（input word）的嵌入层
        self.embed_u = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 中心词（output word）的嵌入层
        self.embed_v = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, contexts, center):
        """
        前向传播函数。

        参数:
        - contexts: 上下文词索引，形状为 (batch_size, max_len)
        - center: 中心词索引，形状为 (batch_size, 1)
        - max_len: 上下文窗口大小
        
        返回:
        - pred: 点积得分，形状为 (batch_size, 1, 1)
        """
        # 获取上下文词的嵌入向量 (batch_size, max_len, embed_size)
        u = self.embed_u(contexts)  # shape: (B, M, E)
        
        # 对上下文词的嵌入向量进行平均 (batch_size, 1, embed_size)
        v = u.mean(dim=1, keepdim=True)  # shape: (B, 1, E)
        
        # 获取中心词的嵌入向量 (batch_size, 1, embed_size)
        center_embed = self.embed_v(center)  # shape: (B, 1, E)
        
        # 计算点积：v @ center_embed^T
        # center_embed.permute(0, 2, 1) -> (B, E, 1)
        pred = torch.bmm(v, center_embed.permute(0, 2, 1))  # shape: (B, 1, 1)
        
        return pred
        # 这个 pred 张量通常会被送入 sigmoid 函数，转化为概率
        # sigmoid(pred) -> (B,1,1)
