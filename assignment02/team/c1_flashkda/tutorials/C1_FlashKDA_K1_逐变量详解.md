# FlashKDA K1 逐变量详解：形状、含义、来源与用途

> 这份文档只讲 K1：`_flash_kda_fwd_prepare`。  
> 目标不是背变量名，而是理解 K1 为什么能把 16 次逐 token 递推，准备成 K2 可以使用的六个矩阵/向量。

---

## 0. 先建立唯一需要记住的总图

固定一个 sequence、一个 head、一个 chunk：

```text
C  = CHUNK       = 16    当前组有 16 个 token
Dk = key/query维 = 128
Dv = value维     = 128
```

K1 的输入：

```text
Q       [16,128]
K       [16,128]
g_raw   [16,128]
beta    [16]
A_log   [1]       当前 head 一个标量
dt_bias [128]     当前 head 每个 key 维度一个 bias
scale   [1]       通常为 1/sqrt(128)
```

K1 不读取：

```text
V       [16,128]
S_in    [128,128]
```

K1 的六个最终输出：

```text
k_decayed   Kd    [16,128]
q_decayed   Qd    [16,128]
k_restored  Kr    [16,128]
g_total     Eend  [128]
INV               [16,16]
Mqk               [16,16]
```

它们会写入 global workspace，随后由 K2 读取。

可以先把六个输出按作用分成三组：

```text
一、旧状态在 chunk 内怎样传播
   Kd、Qd、Eend

二、16 个 token 之间怎样互相影响
   INV、Mqk

三、16 个 token 的写入怎样汇总到 chunk 末尾
   Kr
```

完整数据流：

```text
Q、K ──L2 normalize──→ Qhat、Khat
                           │
g_raw、A_log、dt_bias      │
        │                  │
        ▼                  │
激活 gate γ [16,128]       │
        │                  │
沿 token cumsum            │
        ▼                  │
G [16,128]                 │
        │                  │
        ├──→ Kd [16,128] ──┼──→ L [16,16] ─→ INV [16,16]
        ├──→ Qd [16,128] ──┼──→ Mqk [16,16]
        ├──→ Ki [16,128] ──┘       Ki 只是 K1 临时变量
        ├──→ Kr [16,128]
        └──→ Eend [128]

写入 workspace：Kd、Qd、Kr、Eend、INV、Mqk
```

---

## 1. 为什么 K1 可以不看 V 和 state

逐 token KDA 的概念公式是：

$$
\bar S_i=\operatorname{diag}(e^{\gamma_i})S_{i-1},
$$

$$
u_i=\beta_i\left(v_i-k_i\bar S_i\right),
$$

$$
S_i=\bar S_i+k_i^Tu_i,
$$

$$
o_i=\operatorname{scale}\,q_iS_i.
$$

这里：

- $S_i$ 是状态，数学上记为 `[Dk,Dv]=[128,128]`；
- $u_i$ 是本 token 真正写入的 value 修正量，形状 `[1,Dv]`；
- $\gamma_i$ 是激活后的 log-decay，形状 `[1,Dk]`；
- $\beta_i$ 是一个标量。

如果直接执行，这 16 个 token 必须串行：

```text
S_in → token 0 → S0 → token 1 → S1 → ... → token 15 → S_out
```

但是，token 之间“谁影响谁、衰减多少”的**系数**只由以下变量决定：

```text
Q、K、gate、beta
```

它不依赖具体的：

```text
V、S_in
```

因此 K1 可以提前计算所有系数；K2 拿到真实的 V 和当前 S 后，再代入这些系数。

可以类比解线性方程：

```text
K1：先根据题目结构构造系数矩阵 A
K2：拿到右端项 b 后，计算 x = A^-1 b
```

这就是 K1 叫 `prepare` 的原因。

---

## 2. 输入矩阵到底怎样排列

整个模型输入是：

```text
q、k、v、g：[B,T,H,D]
```

官方形状：

```text
B=1, T=8192, H=96, D=128
```

K1 的一个 CTA 只取：

```text
一个 sequence
一个 head
一个 16-token chunk
```

因此数学视角下，一个 CTA 看到：

```text
             D=128 列
          ┌─────────────┐
token 0   │             │
token 1   │             │
...       │  [16,128]   │
token 15  │             │
          └─────────────┘
            C=16 行
```

三张这样的矩阵分别是 Q、K、g_raw。

源码中的 CUTE `QKLayout`、`MMALayout`、swizzle 只是改变元素在 shared memory 中的物理排列，不改变上面的数学形状。

---

## 3. 第一步：Q、K 做 L2 normalization

对每个 token 的 128 维 q/k 独立归一化：

$$
\widehat q_i=\frac{q_i}{\sqrt{\sum_dq_{i,d}^2+10^{-6}}},
$$

$$
\widehat k_i=\frac{k_i}{\sqrt{\sum_dk_{i,d}^2+10^{-6}}}.
$$

形状不变：

```text
Q     [16,128] → Qhat [16,128]
K     [16,128] → Khat [16,128]
```

为什么需要归一化：

1. 让 q/k 点积主要表达方向相似度，而不是向量长度；
2. 控制 `L`、`Mqk` 的元素范围；
3. 避免递推状态更新因为异常大的 key norm 失稳；
4. 给 FP16/BF16 小矩阵计算提供更可控的数值范围。

源码位置：`fwd_kernel1.cuh:264` 开始。

---

## 4. 第二步：从 raw gate 得到真正的 log-decay

raw gate 不是直接拿来做指数。对当前 head 的 token i、维度 d：

$$
\gamma_{i,d}
=\text{lower\_bound}\cdot
\sigma\!\left(
e^{A_{log,h}}
(g_{raw,i,d}+dt\_bias_{h,d})
\right).
$$

当 `lower_bound=-5` 时：

$$
\gamma_{i,d}\in[-5,0].
$$

每个维度的实际衰减因子是：

$$
d_{i,d}=e^{\gamma_{i,d}}\in[e^{-5},1].
$$

因此 gate 不是一个“开/关”布尔量，而是 128 个独立的遗忘比例：

```text
token i 的 gate：

[维度0衰减, 维度1衰减, ..., 维度127衰减]

形状：[128]
```

16 个 token 合起来：

```text
gamma [16,128]
```

### 4.1 源码为什么用 `ex2` 而不是 `exp`

内核把自然指数的 log-decay 换成 base-2：

$$
\widetilde\gamma=\frac{\gamma}{\ln2}.
$$

于是：

$$
2^{\widetilde\gamma}=e^\gamma.
$$

`flash_kda.cpp` 中：

```cpp
gate_scale = lower_bound * 1.4426950408889634;
```

其中 `1.442695...=1/ln(2)`。

后面的源码变量 `g`/`g_total` 实际使用 base-2 坐标，并调用 `ex2.approx.ftz.f32`。为了让数学含义直观，本文后面仍使用自然指数 $e^G$ 表示；两者数学等价。

源码位置：

- `fwd_kernel1.cuh:256`：计算 `exp(A_log[h])`；
- `fwd_kernel1.cuh:305`：gate activation 与 cumsum；
- `flash_kda.cpp:128`：换底系数。

---

## 5. 第三步：沿 token 方向累计 gate

对 chunk 内第 i 个 token：

$$
G_i=\gamma_0+\gamma_1+\cdots+\gamma_i.
$$

因为每个 $\gamma_i$ 都有 128 个维度，所以：

```text
G [16,128]
```

展开来看：

```text
G[0]  = gamma[0]                              [128]
G[1]  = gamma[0] + gamma[1]                   [128]
G[2]  = gamma[0] + gamma[1] + gamma[2]        [128]
...
G[15] = gamma[0] + ... + gamma[15]            [128]
```

定义累计衰减矩阵：

$$
E_i=e^{G_i}.
$$

那么：

```text
E [16,128]
```

E 的第 i 行表示：

> 从 chunk 入口开始，旧状态连续经过 token 0、1、…、i 的 gate 后，各个 key 维度还剩多少。

最后一行：

$$
E_{end}=e^{G_{15}}in\mathbb{R}^{128}
$$

就是 K1 最终输出的 `g_total [128]`。

注意源码名字叫 `g_total`，但写入 workspace 前已经对累计 log-gate 做过 `ex2`，所以传给 K2 的是**总衰减因子**，不是原始 log-gate。

---

## 6. 为什么要同时构造正指数和负指数

如果 token j 写入状态，token i>j 查询这次写入，它中间经历的衰减是：

$$
e^{G_i-G_j}.
$$

把它拆开：

$$
e^{G_i-G_j}=e^{G_i}e^{-G_j}.
$$

这正是 K1 同时构造：

```text
正累计指数 e^(G_i)
负累计指数 e^(-G_j)
```

的根本原因。

这样“任意两个 token 的相对衰减”就能通过一次普通点积获得，而不需要为每个 `(i,j,d)` 单独计算指数差。

这是理解 `k_decayed` 和 `k_inv` 的钥匙。

---

## 7. `k_decayed`：旧状态对每个 token 的预测坐标

数学定义：

$$
K_d[i]=\widehat k_i\odot e^{G_i}.
$$

形状：

```text
Khat         [16,128]
exp(G)       [16,128]
k_decayed    [16,128]
```

`⊙` 表示逐元素乘法。

### 7.1 为什么 K2 需要它

设进入当前 chunk 的状态是：

```text
S_in [Dk,Dv] = [128,128]
```

旧状态从 chunk 入口传播到第 i 个 token 后，是：

$$
\operatorname{diag}(e^{G_i})S_{in}.
$$

token i 用 key 查询它：

$$
\widehat k_i\operatorname{diag}(e^{G_i})S_{in}.
$$

把 gate 合并到 key：

$$
\left(\widehat k_i\odot e^{G_i}\right)S_{in}
=K_d[i]S_{in}.
$$

16 个 token 一起写成：

$$
P=K_dS_{in}.
$$

形状：

```text
k_decayed [16,128]
S_in     [128,128]
P        [16,128]
```

所以 `k_decayed` 的作用是：

> 让 K2 用一个 `m16n128k128` GEMM，同时得到输入状态对 16 个 token 的基础 value 预测。

P 还没有包含当前 chunk 内更早 token 的写入；这部分稍后由 `INV` 修正。

---

## 8. `q_decayed`：旧状态对输出的贡献坐标

数学定义：

$$
Q_d[i]=\operatorname{scale}\cdot\widehat q_i\odot e^{G_i}.
$$

形状：

```text
Qhat         [16,128]
exp(G)       [16,128]
q_decayed    [16,128]
```

和 `k_decayed` 的逻辑相同，只是用途不同：

- `k_decayed` 用来计算 value prediction；
- `q_decayed` 用来计算 output。

K2 计算输入状态贡献：

$$
O_{old}=Q_dS_{in}.
$$

形状：

```text
q_decayed [16,128]
S_in     [128,128]
O_old    [16,128]
```

`O_old[i]` 表示：

> 第 i 个 token 从进入 chunk 之前就存在的旧状态中读到的输出。

它还不包括当前 chunk 内 token 0…i 的新写入；那部分由 `Mqk @ U` 加上。

---

## 9. `k_inv`：制造任意两个 token 的相对衰减

数学定义：

$$
K_{inv}[j]=\widehat k_j\odot e^{-G_j}.
$$

形状：

```text
k_inv [16,128]
```

它只在 K1 的 shared memory 中使用，不写入 workspace。

### 9.1 `k_decayed[i] · k_inv[j]` 的含义

$$
K_d[i]K_{inv}[j]^T
=\sum_d
\widehat k_{i,d}\widehat k_{j,d}
e^{G_{i,d}}e^{-G_{j,d}}.
$$

合并指数：

$$
=\sum_d
\widehat k_{i,d}\widehat k_{j,d}
e^{G_{i,d}-G_{j,d}}.
$$

当 i>j 时：

$$
G_i-G_j=\gamma_{j+1}+\cdots+\gamma_i.
$$

这表示 token j 的写入传播到 token i 时经历的 gate。

因此这个点积同时包含：

1. key i 与 key j 的相似程度；
2. token j→token i 之间的逐维衰减。

`k_inv` 可以理解成一个“抵消 token j 之前累计 gate 的坐标变换器”。

---

## 10. `k_restored`：把每个 token 的写入搬到 chunk 末尾

数学定义：

$$
K_r[i]=\widehat k_i\odot e^{G_{15}-G_i}.
$$

形状：

```text
k_restored [16,128]
```

为什么叫 restored：源码先构造 $\widehat k_i e^{-G_i}$，再乘 chunk 总衰减 $e^{G_{15}}$：

$$
\widehat k_i e^{-G_i}e^{G_{15}}
=\widehat k_i e^{G_{15}-G_i}.
$$

### 10.1 每一行的含义

```text
Kr[0]  = k0  × exp(gamma1 + ... + gamma15)
Kr[1]  = k1  × exp(gamma2 + ... + gamma15)
...
Kr[14] = k14 × exp(gamma15)
Kr[15] = k15
```

token 0 的写入距离 chunk 末尾最远，所以衰减最多；token 15 已经在末尾，不再经历后续 gate。

K2 最后计算：

$$
\Delta S=K_r^TU.
$$

形状：

```text
k_restored.T [128,16]
U            [16,128]
delta_state  [128,128]
```

所以 `k_restored` 的作用是：

> 把 16 个 token 分别产生的状态写入，全部换算到 chunk 结束时刻，再一次性合并成 `[128,128]` 的状态增量。

---

## 11. `g_total`：把输入状态整体搬到 chunk 末尾

数学定义：

$$
E_{end}=e^{G_{15}}.
$$

形状：

```text
g_total [128]
```

为什么不是一个标量：KDA 是 per-dimension gate，128 个 key 维度各有不同的衰减。

K2 使用它计算：

$$
S_{old,end}=\operatorname{diag}(E_{end})S_{in}.
$$

形状：

```text
diag(g_total) [128,128]
S_in         [128,128]
结果         [128,128]
```

源码不会真的构造 `diag(g_total)`，而是逐元素缩放 state 对应的 key 维度。

最终 chunk 状态为：

$$
S_{out}
=\operatorname{diag}(E_{end})S_{in}
+K_r^TU.
$$

两项含义：

```text
旧状态经过整个 chunk 后剩下的部分
                +
当前 16 个 token 新写入并传播到末尾的部分
```

---

## 12. `L`：记录 token 写入对后续 residual 的影响

K1 先计算：

$$
L_{raw}=K_dK_{inv}^T.
$$

矩阵形状：

```text
k_decayed [16,128]
k_inv.T   [128,16]
L_raw     [16,16]
```

对应 GEMM：

```text
m16n16k128
```

`L_raw[i,j]` 表示 token j 写入的 key 方向传播到 token i 后，与 token i 的 key 有多强耦合。

### 12.1 为什么只保留严格下三角

时间因果关系要求：

```text
token 0 不可能受 token 1 影响
token 1 可以受 token 0 影响
token 2 可以受 token 0、1 影响
```

所以最终 L 为：

$$
L_{i,j}=
\begin{cases}
\beta_i\,K_d[i]K_{inv}[j]^T,&j<i,\\
0,&j\ge i.
\end{cases}
$$

形状：

```text
L [16,16]
```

结构：

```text
        来源 token j
        0  1  2  3  ...
目标 0 [0  0  0  0  ...]
token1 [x  0  0  0  ...]
i    2 [x  x  0  0  ...]
     3 [x  x  x  0  ...]
```

### 12.2 为什么 L 的第 i 行乘 beta_i

token i 的写入修正是：

$$
u_i=\beta_i(v_i-\text{prediction}_i).
$$

早期 token 对 token i prediction 的影响，也位于这个括号内，因此整体会乘当前 token 的 $\beta_i$。

所以 L 的 row i 乘的是 $\beta_i$，不是来源 token j 的 $\beta_j$。

来源 token 的 beta 已经包含在它自己的 $u_j$ 中。

### 12.3 为什么对角线是零

在 token i 计算 residual 时，它自己的新写入还没有发生，所以不能让 $u_i$ 反过来影响自己的 residual。

因此 L 是严格下三角，不包含对角线。

源码位置：

- `fwd_kernel1.cuh:481`：`L` 与 `Mqk` GEMM；
- `fwd_kernel1.cuh:492`：因果 mask、beta 与 `I-L` 初始化。

---

## 13. `INV`：一次解决 16 个 residual 的因果依赖

K2 会先根据输入状态计算一个“尚未考虑 chunk 内更新”的基础 residual：

$$
R_i=\beta_i\left(v_i-K_d[i]S_{in}\right).
$$

合成矩阵：

```text
R [16,128]
```

但真正的 $u_i$ 还要减去前面 token 写入对当前 prediction 的贡献：

$$
u_i+\sum_{j<i}L_{i,j}u_j=R_i.
$$

16 行一起写成：

$$
(I+L)U=R.
$$

其中：

```text
I+L [16,16]
U   [16,128]
R   [16,128]
```

因此：

$$
U=(I+L)^{-1}R.
$$

K1 预先生成：

$$
INV=(I+L)^{-1}.
$$

形状：

```text
INV [16,16]
```

K2 只需要：

```text
INV [16,16] @ R [16,128]
→ U [16,128]
```

### 13.1 为什么能用有限 Neumann 级数

L 是 16×16 严格下三角，所以：

$$
L^{16}=0.
$$

因此：

$$
(I+L)^{-1}
=I-L+L^2-L^3+\cdots-L^{15}.
$$

源码使用乘积分解：

$$
(I+L)^{-1}
=(I-L)(I+L^2)(I+L^4)(I+L^8).
$$

实现过程：

```text
初值 INV = I - L

计算 L²
INV = INV + INV @ L²

计算 L⁴
INV = INV + INV @ L⁴

计算 L⁸
INV = INV + INV @ L⁸
```

总共 6 个 `m16n16k16` FP16 MMA：

```text
L×L
INV×L²
L²×L²
INV×L⁴
L⁴×L⁴
INV×L⁸
```

最终 `INV` 转成 BF16，写入 workspace。

源码位置：

- `utils.cuh:190`：Neumann helper；
- `fwd_kernel1.cuh:513`：调用 helper。

---

## 14. `Mqk`：当前 chunk 的写入怎样影响每个输出

K1 计算：

$$
M_{qk,raw}=Q_dK_{inv}^T.
$$

形状：

```text
q_decayed [16,128]
k_inv.T   [128,16]
Mqk_raw   [16,16]
```

对应 GEMM：

```text
m16n16k128
```

`Mqk[i,j]` 表示：

> token j 的状态写入 $u_j$ 传播到 token i 后，会被 query i 读出多少。

### 14.1 为什么 Mqk 保留下三角和对角线

第 i 个 token 的输出是在 token i 更新状态之后计算的，所以它能看到：

```text
token 0 的输出：看到 u0
token 1 的输出：看到 u0、u1
token 2 的输出：看到 u0、u1、u2
```

因此：

$$
M_{qk}[i,j]=
\begin{cases}
Q_d[i]K_{inv}[j]^T,&j\le i,\\
0,&j>i.
\end{cases}
$$

结构是包含对角线的下三角：

```text
[x  0  0  0]
[x  x  0  0]
[x  x  x  0]
[x  x  x  x]
```

这和 L 不同：

| 矩阵 | 对角线 | 原因 |
|---|---|---|
| L | 0 | token i 算 residual 时，还没写入自己 |
| Mqk | 保留 | token i 算 output 时，已经写入自己 |

### 14.2 为什么 Mqk 不乘 beta

因为 K2 中：

$$
U=INV\left(\beta\odot(V-P)\right).
$$

beta 已经进入 U。`Mqk @ U` 再乘一次 beta 就会重复。

### 14.3 K2 怎样使用 Mqk

输入状态贡献：

$$
O_{old}=Q_dS_{in}.
$$

当前 chunk 写入贡献：

$$
O_{intra}=M_{qk}U.
$$

最终：

$$
O=O_{old}+O_{intra}.
$$

形状：

```text
Qd @ S_in    → [16,128]
Mqk @ U      → [16,128]
相加         → [16,128]
```

---

## 15. 用两个 token 手算 K1：这是理解所有变量的关键

把 `CHUNK=16` 临时缩小成 `C=2`。仍保留 128 维向量，但只写两个 token。

### 15.1 累计 gate

$$
G_0=\gamma_0,
$$

$$
G_1=\gamma_0+\gamma_1.
$$

### 15.2 四种 key/query 变体

$$
K_d[0]=\widehat k_0e^{\gamma_0},
$$

$$
K_d[1]=\widehat k_1e^{\gamma_0+\gamma_1}.
$$

$$
K_{inv}[0]=\widehat k_0e^{-\gamma_0},
$$

$$
K_{inv}[1]=\widehat k_1e^{-(\gamma_0+\gamma_1)}.
$$

$$
K_r[0]=\widehat k_0e^{\gamma_1},
$$

$$
K_r[1]=\widehat k_1.
$$

$$
Q_d[i]=\operatorname{scale}\,\widehat q_i e^{G_i}.
$$

这里的指数都是 128 维逐元素指数/乘法。

### 15.3 L

只有 token 0 能影响 token 1：

$$
L=
\begin{bmatrix}
0&0\\
\ell_{10}&0
\end{bmatrix}.
$$

其中：

$$
\ell_{10}
=\beta_1K_d[1]K_{inv}[0]^T.
$$

展开指数：

$$
\ell_{10}
=\beta_1
\sum_d
\widehat k_{1,d}\widehat k_{0,d}e^{\gamma_{1,d}}.
$$

注意 $\gamma_0$ 消掉了，只剩 token 0 写入到 token 1 之间经历的 $\gamma_1$。这正是相对衰减。

### 15.4 INV

$$
I+L=
\begin{bmatrix}
1&0\\
\ell_{10}&1
\end{bmatrix}.
$$

所以：

$$
INV=(I+L)^{-1}
=
\begin{bmatrix}
1&0\\
-\ell_{10}&1
\end{bmatrix}.
$$

K2 的基础 residual：

$$
R=
\begin{bmatrix}
R_0\\
R_1
\end{bmatrix}.
$$

于是：

$$
U=INV\,R
=
\begin{bmatrix}
R_0\\
R_1-\ell_{10}R_0
\end{bmatrix}.
$$

这说明：

- token 0 没有前驱，$U_0=R_0$；
- token 1 从基础 residual 中减去 token 0 已经造成的影响。

### 15.5 Mqk

$$
M_{qk}=
\begin{bmatrix}
m_{00}&0\\
m_{10}&m_{11}
\end{bmatrix}.
$$

其中：

$$
m_{00}=\operatorname{scale}\,\widehat q_0\widehat k_0^T,
$$

$$
m_{10}=\operatorname{scale}
\sum_d\widehat q_{1,d}\widehat k_{0,d}e^{\gamma_{1,d}},
$$

$$
m_{11}=\operatorname{scale}\,\widehat q_1\widehat k_1^T.
$$

因此输出的 chunk 内新增部分：

$$
O_{intra,0}=m_{00}U_0,
$$

$$
O_{intra,1}=m_{10}U_0+m_{11}U_1.
$$

token 1 看到 token 0 和自己的写入；token 0 看不到未来的 token 1。

### 15.6 chunk 末状态

旧状态经过两个 gate：

$$
S_{old,end}=\operatorname{diag}(e^{\gamma_0+\gamma_1})S_{in}.
$$

两个 token 的写入传播到末尾：

$$
\Delta S
=\left(\widehat k_0e^{\gamma_1}\right)^TU_0
+\widehat k_1^TU_1.
$$

因此：

$$
S_{out}=S_{old,end}+\Delta S.
$$

这两-token 示例中，K1 的所有变量已经全部出现。

---

## 16. K1 六个输出进入 K2 后分别在哪里用

| K1 输出 | 数学形状 | K2 使用公式 | 得到的形状 |
|---|---:|---|---:|
| `k_decayed` | `[16,128]` | $P=K_dS_{in}$ | `[16,128]` |
| `q_decayed` | `[16,128]` | $O_{old}=Q_dS_{in}$ | `[16,128]` |
| `INV` | `[16,16]` | $U=INV\,[\beta\odot(V-P)]$ | `[16,128]` |
| `Mqk` | `[16,16]` | $O_{intra}=M_{qk}U$ | `[16,128]` |
| `k_restored` | `[16,128]` | $\Delta S=K_r^TU$ | `[128,128]` |
| `g_total` | `[128]` | $S_{old,end}=\operatorname{diag}(E_{end})S_{in}$ | `[128,128]` |

K2 的三个最终公式是：

$$
U=INV\left[\beta\odot(V-K_dS_{in})\right],
$$

$$
O=Q_dS_{in}+M_{qk}U,
$$

$$
S_{out}=\operatorname{diag}(E_{end})S_{in}+K_r^TU.
$$

如果这三个公式能看懂，K1 的变量就不再是六个孤立名字，而是各自占据公式中的一个位置。

---

## 17. 哪些是 K1 临时变量，哪些真的写到 workspace

| 变量 | 形状 | dtype/概念 | 是否写 workspace | 原因 |
|---|---:|---|---|---|
| normalized Q | `[16,128]` | BF16 | 否 | 已折入 Qd |
| normalized K | `[16,128]` | BF16 | 否 | 已折入 Kd/Kr |
| cumulative G | `[16,128]` | FP32 log2 坐标 | 否 | 已折入各种指数变量 |
| `k_inv` | `[16,128]` | BF16 | 否 | 只用于生成 L、Mqk、Kr |
| L | `[16,16]` | FP16 | 否 | K2 只需要它的逆 INV |
| `k_decayed` | `[16,128]` | BF16 | **是** | K2 计算 P |
| `q_decayed` | `[16,128]` | BF16 | **是** | K2 计算旧状态输出 |
| `k_restored` | `[16,128]` | BF16 | **是** | K2 更新状态 |
| `g_total` | `[128]` | FP32 衰减因子 | **是** | K2 衰减旧状态 |
| INV | `[16,16]` | BF16，内部 FP16 求逆 | **是** | K2 解 residual 依赖 |
| Mqk | `[16,16]` | BF16 | **是** | K2 计算 chunk 内输出 |

每个 `(head,chunk)` 的 workspace 大小：

```text
k_decayed   16×128×2 B = 4096 B
q_decayed   16×128×2 B = 4096 B
k_restored  16×128×2 B = 4096 B
g_total        128×4 B =  512 B
INV          16×16×2 B =  512 B
Mqk          16×16×2 B =  512 B
--------------------------------
合计                     13824 B
                         = 13.5 KiB
```

这 13.5 KiB 会被 K1 写一次、K2 读一次。

---

## 18. K1 的实际执行顺序与源码对应

```text
阶段 1：TMA load
  q、k、g_raw、beta、dt_bias → shared memory

阶段 2：Q/K normalization
  Q、K → Qhat、Khat
  源码约 fwd_kernel1.cuh:264

阶段 3：gate activation + cumsum
  g_raw → gamma → G
  生成 Gend
  源码约 fwd_kernel1.cuh:305

阶段 4：decay_apply
  Qhat、Khat、G
    → q_decayed
    → k_decayed
    → k_inv
    → k_restored
    → g_total
  源码约 fwd_kernel1.cuh:359

阶段 5：两个 GEMM
  Kd @ Ki.T → L_raw       [16,16]
  Qd @ Ki.T → Mqk_raw     [16,16]
  源码约 fwd_kernel1.cuh:481

阶段 6：causal mask + beta
  L 只保留严格下三角并按行乘 beta
  Mqk 保留下三角和对角线
  初始化 INV = I - L
  源码约 fwd_kernel1.cuh:492

阶段 7：Neumann inverse
  INV = (I+L)^-1
  源码约 fwd_kernel1.cuh:513

阶段 8：TMA store workspace
  Kd、Qd、Kr、g_total、INV、Mqk
  源码约 fwd_kernel1.cuh:519—588
```

### 18.1 为什么源码先写 `INV = I - L`

这是 Neumann 乘积分解的第一因子：

$$
(I+L)^{-1}=(I-L)(I+L^2)(I+L^4)(I+L^8).
$$

所以 `I-L` 只是 inverse 计算的初值，不代表最终求的是 `(I-L)^-1`。

---

## 19. K1 中真正的矩阵乘法有哪些

固定一个 head、一个 chunk：

### 19.1 构造 L

```text
[16,128] @ [128,16] → [16,16]

M=16, N=16, K=128
```

### 19.2 构造 Mqk

```text
[16,128] @ [128,16] → [16,16]

M=16, N=16, K=128
```

### 19.3 求 INV

6 次：

```text
[16,16] @ [16,16] → [16,16]

M=16, N=16, K=16
```

因此 K1 不是一个 `96×128 @ 128×...` 的大 GEMM，而是：

```text
每个 (head,chunk) 独立执行：

2 个 m16n16k128
6 个 m16n16k16
```

官方固定形状：

```text
512 chunks/head × 96 heads = 49152 个 K1 CTA
```

这些 CTA 之间不共享 state，因此可以高度并行。

---

## 20. 为什么不同 chunk 的累计 gate 可以各自从零开始

这通常是理解 K1 时的一个疑问：

> chunk 1 的 G 为什么不需要加上 chunk 0 的所有 gate？

因为 chunk 1 接收到的 $S_{in}$ 已经是 chunk 0 处理后的状态。

对两个 chunk：

```text
K1：
  chunk 0 在自己的局部坐标中生成 W0
  chunk 1 在自己的局部坐标中生成 W1
  两者可并行

K2：
  S0 + W0 → S1
  S1 + W1 → S2
```

chunk 0 之前的历史被压缩在 S0；chunk 0 的历史被压缩在 S1。每个 chunk 的局部 G 只需要描述“进入本 chunk 后又衰减了多少”。

因此：

- K1 可以独立准备所有 chunk；
- K2 必须按顺序把 $S_c$ 传给 $S_{c+1}$。

---

## 21. 最容易混淆的十个问题

### 21.1 `g_total` 是 scalar 吗？

不是，是 `[128]`。KDA 对 state 的 128 个 key 维度分别衰减。

### 21.2 `g_total` 是 log-gate 吗？

写到 workspace 时不是。它已经是 $e^{G_{15}}$ 的 FP32 衰减因子。

### 21.3 `k_inv` 是矩阵逆吗？

不是。名字中的 inv 指乘了 $e^{-G_i}$。真正的小矩阵逆是 `INV [16,16]`。

### 21.4 L 为什么是 `[16,16]`？

行表示“被影响的 token i”，列表示“影响来源 token j”。一个 chunk 有 16 个 token，因此是 16×16。

### 21.5 INV 为什么也是 `[16,16]`？

它要解的是 16 个 token 之间的三角依赖，不是在 key 的 128 维上求逆。

### 21.6 Mqk 为什么是 `[16,16]`？

它记录 16 个 query 与 16 个状态写入之间的读取权重。

### 21.7 L 和 Mqk 有什么区别？

- L：key→key，修正 residual；
- Mqk：query→key，生成 output。

### 21.8 为什么 L 不保留对角线，Mqk 保留？

- residual 在本 token 写入之前计算，所以 L 不含自己；
- output 在本 token 写入之后计算，所以 Mqk 包含自己。

### 21.9 beta 为什么在 K1、K2 都出现？

- K1 中 beta 进入 L，描述当前 token 的 residual 对前驱影响有多敏感；
- K2 中 beta 进入 $R=\beta(V-P)$，控制实际写入强度。

这是同一个递推公式展开后出现在两个位置，不是错误地重复乘同一项。Mqk 本身不再乘 beta，因为 U 已包含它。

### 21.10 K1 为什么不直接输出 L？

K2 真正需要的是解：

$$
U=(I+L)^{-1}R.
$$

所以 K1 直接把 `INV` 算好，避免 K2 的串行 recurrence 临界路径再做求逆。

---

## 22. 阅读源码时怎样把物理 layout 和数学矩阵分开

源码里会看到：

```cpp
QKLayout
MMALayout
LMLayout
TransposedLMLayout
TMAQKLayout
TMAVOLayout
```

这些名字回答的是：

```text
元素放在 shared memory 的哪个地址
ldmatrix 应该怎样读取
Tensor Core 把哪部分解释成 A/B
TMA 怎样写成消费者所需布局
```

它们没有改变数学意义：

```text
Q/K/Kd/Qd/Ki/Kr 仍然是 [16,128]
L/INV/Mqk         仍然是 [16,16]
g_total           仍然是 [128]
```

建议读源码时先在纸上只写数学 shape；等公式理解后，再看 layout 如何实现这些 shape。

---

## 23. 一页变量字典

| 源码名 | 数学符号 | 形状 | 一句话含义 |
|---|---|---:|---|
| `q_tile` | $\widehat Q$ | `[16,128]` | L2 normalized query |
| `k_tile` | $\widehat K$ | `[16,128]` | L2 normalized key |
| `g_tile` | $G$ | `[16,128]` | chunk 内累计 log-decay |
| `g_total` | $E_{end}$ | `[128]` | 整个 chunk 的逐维衰减因子 |
| `k_decayed` | $K_d$ | `[16,128]` | 从 chunk 输入状态预测 value |
| `q_decayed` | $Q_d$ | `[16,128]` | 从 chunk 输入状态读取 output |
| `k_inv` | $K_{inv}$ | `[16,128]` | 构造 token-token 相对衰减 |
| `k_restored` | $K_r$ | `[16,128]` | 把 token 写入传播到 chunk 末尾 |
| `L` | $L$ | `[16,16]` | 前面 token 对后面 residual 的影响 |
| `INV` | $(I+L)^{-1}$ | `[16,16]` | 并行解开 16-token residual 依赖 |
| `Mqk` | $M_{qk}$ | `[16,16]` | chunk 内写入对各 token 输出的影响 |

---

## 24. 学完 K1 后的自测

### 问题 1

为什么 `k_decayed @ S_in` 能表示旧状态传播到不同 token 后的 prediction？

答案：因为 `k_decayed[i]` 已把从 chunk 入口到 token i 的累计逐维衰减 $e^{G_i}$ 合并进 key。

### 问题 2

为什么需要 `k_inv`？

答案：`e^{G_i}e^{-G_j}=e^{G_i-G_j}`，它让一次矩阵乘得到任意 token j→i 的相对衰减与 key 相似度。

### 问题 3

L 为什么严格下三角？

答案：token j 只能影响未来 token i>j；当前 token 算 residual 时自己的写入还没发生。

### 问题 4

INV 为什么能消除 chunk 内串行 residual 更新？

答案：16 个更新满足三角系统 `(I+L)U=R`，预计算 inverse 后可用一次矩阵乘同时求出所有 U。

### 问题 5

Mqk 与 L 为什么都来自某个 `[16,128] @ [128,16]`，含义却不同？

答案：L 的左操作数是 key，描述写入对未来 prediction/residual 的影响；Mqk 的左操作数是 query，描述这些写入对 output 的读取权重。

### 问题 6

为什么 `k_restored[15]=k[15]`？

答案：最后一个 token 的写入已经位于 chunk 末尾，不再经历后续 gate。

### 问题 7

K1 为什么能让 512 个 chunk 并行？

答案：它只构造局部系数，不读取递推 state；跨 chunk 的历史由 K2 的输入 state 承担。

---

## 25. 最终只背这五句话

```text
1. G 把每个 token 放到“从 chunk 入口累计衰减了多少”的时间坐标中。

2. Kd/Qd 用于读取进入 chunk 的旧状态；Ki 用于算两个 token 之间的相对衰减。

3. L 描述早期 token 写入怎样改变后续 residual，INV 一次解开这条三角依赖。

4. Mqk 描述当前 chunk 的写入怎样被每个 query 读成输出。

5. g_total 衰减旧状态，Kr 把所有新写入传播到 chunk 末尾，两者一起生成下一个 state。
```

对应 K2 的三行公式：

$$
U=INV\left[\beta\odot(V-K_dS_{in})\right],
$$

$$
O=Q_dS_{in}+M_{qk}U,
$$

$$
S_{out}=\operatorname{diag}(g_{total})S_{in}+K_r^TU.
$$

理解这三行后，K1 的每个输出都能在 K2 中找到唯一、明确的位置。
