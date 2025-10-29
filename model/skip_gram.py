import torch
import torch.nn as nn

# 跳元模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        # 中心词（input word）的嵌入层
        self.embed_v = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 上下文词（output word）的嵌入层（也用于负采样词）
        self.embed_u = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, center, contexts_and_negatives):
        """
        前向传播函数。

        参数:
        - center: 中心词索引，形状为 (batch_size, 1)
        - contexts_and_negatives: 上下文词与负采样词索引，形状为 (batch_size, max_len)
        - max_len: 上下文窗口大小 + 负采样样本数
        
        返回:
        - pred: 点积得分，形状为 (batch_size, 1, max_len)
        """
        # 获取中心词的嵌入向量 (batch_size, 1, embed_size)
        v = self.embed_v(center)  # shape: (B, 1, E)

        # 获取上下文词和负采样词的嵌入向量 (batch_size, max_len, embed_size)
        u = self.embed_u(contexts_and_negatives)  # shape: (B, M, E)

        # 计算点积：v @ u^T
        # u.permute(0, 2, 1) -> (B, E, M)
        pred = torch.bmm(v, u.permute(0, 2, 1))  # shape: (B, 1, M)
        
        return pred
        # 这个 pred 张量通常会被送入 sigmoid 函数，转化为概率
        # sigmoid(pred) -> (B,1,M)