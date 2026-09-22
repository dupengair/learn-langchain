*Thinking...*

> The user wants me to modify their MHA implementation to GQA. Let me carefully adapt their code, keeping the same style and adding clear comments where changes are made.
> 
> Key changes for GQA:
> 1. Introduce `num_kv_heads` (fewer than `num_heads` for Q)
> 2. K and V linear layers project to smaller dimension (`num_kv_heads * head_dim` instead of `num_heads * head_dim`)
> 3. After splitting heads, K and V need to be expanded/repeated to match Q's head count
> 4. Everything else stays the same
> 
> Let me adapt their code. Their current setup:
> - batch_size=2, seq_len=3, feature_dim=4


```python
# 分组查询注意力 GQA（从 MHA 改造而来，修改处有 ★ 标记）
import torch
import torch.nn.functional as F

x = torch.randn(2, 3, 4)  # 形状 (batch_size, seq_len, feature_dim)

# ===================== ★ 改动1：新增 num_kv_heads =====================
# MHA 中 Q、K、V 的头数都是 num_heads
# GQA 中 Q 的头数仍为 num_heads，但 K、V 的头数减少为 num_kv_heads
# 要求 num_heads 能被 num_kv_heads 整除，这样每 num_heads//num_kv_heads 个 Q 头共享一组 KV
num_heads = 2       # Q 的头数，和 MHA 一样
num_kv_heads = 1    # ★ K、V 的头数（比 num_heads 小）
head_dim = 2
n_rep = num_heads // num_kv_heads  # ★ 每组 KV 被多少个 Q 头共享（这里 = 2）

feature_dim = num_heads * head_dim          # Q 的总维度 = 4，和 MHA 一样
kv_dim = num_kv_heads * head_dim            # ★ K、V 的总维度 = 2，比 MHA 小

assert x.size(-1) == feature_dim

# ===================== ★ 改动2：K、V 的线性层输出维度变小 =====================
# MHA 中三个线性层输出维度都是 feature_dim(=4)
# GQA 中 K、V 的线性层输出维度变成 kv_dim(=2)，参数量减少
linear_q = torch.nn.Linear(feature_dim, feature_dim)   # 输出维度 4，和 MHA 一样
linear_k = torch.nn.Linear(feature_dim, kv_dim)        # ★ 输出维度 2（MHA 中是 4）
linear_v = torch.nn.Linear(feature_dim, kv_dim)        # ★ 输出维度 2（MHA 中是 4）

Q = linear_q(x)  # 形状 (batch_size, seq_len, feature_dim)  即 (2,3,4)
K = linear_k(x)  # ★ 形状 (batch_size, seq_len, kv_dim)     即 (2,3,2)，比 MHA 小
V = linear_v(x)  # ★ 形状 (batch_size, seq_len, kv_dim)     即 (2,3,2)，比 MHA 小

# 分头函数和 MHA 一样，只是传入的 num_heads 参数不同
def split_heads(tensor, num_heads):
    batch_size, seq_len, dim = tensor.size()
    head_dim = dim // num_heads
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return output

Q = split_heads(Q, num_heads)      # 形状 (2, 2, 3, 2) — 2 个 Q 头，和 MHA 一样
K = split_heads(K, num_kv_heads)   # ★ 形状 (2, 1, 3, 2) — 只有 1 个 K 头（MHA 中是 2 个）
V = split_heads(V, num_kv_heads)   # ★ 形状 (2, 1, 3, 2) — 只有 1 个 V 头（MHA 中是 2 个）

# ===================== ★ 改动3（GQA 的核心）：扩展 KV 头数以匹配 Q =====================
# MHA 中不需要这一步，因为 Q、K、V 头数相同，可以直接算注意力
# GQA 中 K、V 只有 num_kv_heads 个头，需要通过 repeat 扩展到 num_heads 个
# 这样每 n_rep 个 Q 头就会共享同一份 KV，实现"分组"效果
def repeat_kv(x, n_rep):
    """将 KV 的头从 num_kv_heads 扩展到 num_heads"""
    if n_rep == 1:
        return x  # 如果 num_kv_heads == num_heads，退化为 MHA，无需扩展
    batch_size, num_kv_heads, seq_len, head_dim = x.size()
    # unsqueeze(2) 在头维度后插入一个新维度，然后 expand 复制 n_rep 次，最后 reshape 合并
    # (B, num_kv_heads, 1, seq_len, head_dim)
    # -> (B, num_kv_heads, n_rep, seq_len, head_dim)
    # -> (B, num_kv_heads * n_rep, seq_len, head_dim) = (B, num_heads, seq_len, head_dim)
    return (
        x.unsqueeze(2)
         .expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
         .reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    )

K = repeat_kv(K, n_rep)  # ★ 形状从 (2,1,3,2) 扩展为 (2,2,3,2)，现在和 Q 头数一致
V = repeat_kv(V, n_rep)  # ★ 形状从 (2,1,3,2) 扩展为 (2,2,3,2)

# ===================== 以下和 MHA 完全一样，无任何改动 =====================

raw_weights = torch.matmul(Q, K.transpose(-2, -1))  # 形状 (batch_size, num_heads, seq_len, seq_len)

scale_factor = K.size(-1) ** 0.5
scaled_weights = raw_weights / scale_factor

attn_weights = F.softmax(scaled_weights, dim=-1)

attn_outputs = torch.matmul(attn_weights, V)  # 形状 (batch_size, num_heads, seq_len, head_dim)

def combine_heads(tensor, num_heads):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim
    output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
    return output

attn_outputs = combine_heads(attn_outputs, num_heads)  # 形状 (batch_size, seq_len, feature_dim)

linear_out = torch.nn.Linear(feature_dim, feature_dim)
attn_outputs = linear_out(attn_outputs)  # 形状 (batch_size, seq_len, feature_dim)

print("加权信息：", attn_outputs)
```

总结一下，相比你的 MHA 代码，**只有三处改动**：

**改动 1** — 新增 `num_kv_heads` 参数。MHA 中 Q/K/V 头数相同，GQA 中 K/V 的头数 `num_kv_heads` 小于 Q 的头数 `num_heads`。这是 GQA 的核心定义。

**改动 2** — K、V 的线性投影层变小了。输出维度从 `num_heads * head_dim` 缩减为 `num_kv_heads * head_dim`，所以参数量更少，KV Cache 占用的显存也更小。

**改动 3** — 在注意力计算之前，用 `repeat_kv` 把 K、V 从 `num_kv_heads` 个头扩展回 `num_heads` 个头。扩展之后 Q/K/V 的 shape 完全对齐，后续的注意力计算和 MHA 一模一样。这个 repeat 操作就是"分组共享"的实现——同一份 KV 数据被复制给了同组内的多个 Q 头。

你可以试试把 `num_kv_heads` 改成和 `num_heads` 相同的值（即 2），会发现 `repeat_kv` 直接返回原张量，整个代码退化回你原来的 MHA，验证 GQA 确实是 MHA 的泛化。