# bert全称：Bidirectional Encoder Representations from Transformers 来自Transformers的双向编码器
# 作用：将输入的文本序列转换为向量表示，用于下游任务

# BERT 是一种基于 Transformer 架构的预训练语言模型，它在预训练时使用了两个主要任务：

# 掩码语言建模（Masked Language Modeling, MLM）
# 下一句预测（Next Sentence Prediction, NSP）

import torch
from torch import nn
from d2l import torch as d2l


# BERT模型
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        # BERT编码器层
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        
        # 用于对 BERT 编码器输出的 [CLS] 标记对应的隐藏状态进行非线性变换
        # 以便后续用于下一句预测任务NSP
        # 对 [CLS] 向量再过一个线性层 + tanh，得到“pooled representation”，再送入 NSP 分类器
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), 
                                    nn.Tanh()) #这是一个激活函数，将线性变换后的结果压缩到(−1,1)区间
        # 任务一:掩蔽语言模型MLM层
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        # 任务二:下一句预测NSP层
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # 编码结果
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        # encoded_X(B,Q,H) -> cls(B,H)
        cls = encoded_X[:, 0, :]
        # cls(B,H) -> cls(B,H)
        cls = self.hidden(cls) # pooled representation:池化表示
        # nsp_Y_hat(B,2)
        nsp_Y_hat = self.nsp(cls)
        # encoded_X(B,Q,H)
        # mlm_Y_hat(B,Q,V)
        # nsp_Y_hat(B,2)
        return encoded_X, mlm_Y_hat, nsp_Y_hat