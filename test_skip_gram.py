import math
import torch
from torch import nn
from d2l import torch as d2l
from model import SkipGramModel
from common import try_gpu, display_model
import random
import os

# 带掩码的二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        # 计算逐元素的 BCE loss（logits 输入）
        loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        
        if mask is not None:
            # 只对 mask > 0 的位置求平均（按样本）
            # 避免除零：当某个样本的 mask 全为 0 时，loss 应为 0
            loss = loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # 如果没有 mask，就直接平均
            loss = loss.mean(dim=1)
        
        return loss



# 训练跳元模型
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    # 初始化模型参数
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    # 定义损失函数
    # loss = nn.BCEWithLogitsLoss(reduction='none')
    loss = SigmoidBCELoss()
    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            # 前向传播
            pred = net(center, context_negative)
            # 计算损失
            l = loss(pred.reshape(label.shape).float(), label.float(), mask)
            # 反向传播
            l.sum().backward()
            # 更新参数
            optimizer.step()
            metric.add(l.sum(), l.numel())
            # 打印每一个 batch 的损失
            print(f'epoch {epoch + 1}, iter {i + 1}, loss {metric[0] / metric[1]:.3f}')
        
        # 打印每个 epoch 的损失
        print(f'epoch {epoch + 1}, loss {metric[0] / metric[1]:.3f}, ' f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')



# 测试:获取与查询词最相似的 k 个词
def get_similar_tokens(vocab, query_token, k, embed):
    # 获取其底层张量（不带梯度信息），赋值给 W
    W = embed.weight.data # 获取嵌入层的权重矩阵，它保存了词汇表中每个词的向量表示
    x = W[vocab[query_token]] # 获取查询词的向量表示

    # 计算 x 与嵌入矩阵中所有词向量的余弦相似度
    # 加上 1e-9 是为了防止除零错误（数值稳定性）
    # mv是矩阵向量乘法，计算 W 中每一行与 x 的点积
    # * 是逐元素乘法
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    # 返回 cos 中最大的 k+1 个值及其索引（包括查询词本身）
    # [1] 表示只取索引（即词在词汇表中的位置）
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


def main():
    # 超参数
    lr = 0.002
    num_epochs = 5

    # 获取可用设备
    device = try_gpu()
    print(f'Using device: {device}')
    
    # 加载数据集
    batch_size = 512
    max_window_size = 5
    num_noise_words = 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)
    vocab_size = len(vocab)
    embed_size = 100

    # 创建模型
    model = SkipGramModel(vocab_size, embed_size).to(device)
    display_model(model)

    # 训练模型
    train(model, data_iter, lr, num_epochs, device)

    # 测试:获取与查询词最相似的 k 个词
    get_similar_tokens(vocab, 'chip', 3, model.embed_v)








def read_ptb():
    """Load the PTB dataset into a list of text lines.

    Defined in :numref:`sec_word2vec_data`"""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

def subsample(sentences, vocab):
    """Subsample high-frequency words.

    Defined in :numref:`sec_word2vec_data`"""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram.

    Defined in :numref:`sec_word2vec_data`"""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        """Defined in :numref:`sec_word2vec_data`"""
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory.

    Defined in :numref:`subsec_word2vec-minibatch-loading`"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab


if __name__ == '__main__':
    main()