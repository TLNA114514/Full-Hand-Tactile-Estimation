# DINOv3 ReZero BaseAnchor 训练集拟合审计

## 1. 审计目标

本报告分析实验 `mixed_dense_v2_dinov3_rezero_baseanchor` 的 RMSE-best checkpoint 在完整 mixed training set 上的拟合情况，重点回答以下问题：

1. 模型是否已经拟合训练集？
2. 为什么训练集 Pred/GT total volume 接近 1，但 V-IoU Frame-Macro 仍只有 0.4824？
3. 误差主要来自背景底噪、高压幅值不足、空间拓扑偏移，还是整帧尺度错误？
4. BaseAnchor/ReZero residual 在训练集上实际做了什么？
5. 训练集与 test split 的差距说明了什么？

本报告不是一次新的模型推理。它使用已有完整 train-fit 报告以及诊断 CSV 进行复核和重组。

## 2. 实验与数据口径

### 2.1 模型

- Experiment: `mixed_dense_v2_dinov3_rezero_baseanchor`
- Checkpoint: `rmse-best`
- Visual backbone: frozen DINOv3 H+/16
- Tactile head: `dense_v2_dino_rezero`
- Feature layers: block 8/16/24/32
- Base anchor weight: `0.5`
- BBox rescale factor: `2.0`
- Train sampling: frame-uniform
- Dense output: 13,614 subdiv MANO vertices
- Eval/loss palm mask: `group_negative`，共 6,623 个 palm vertices

### 2.2 数据规模

- 完整 train frames: **557,293**
- OpenTouch train samples: 127,326
- TouchAnything train samples: 429,967
- 完整 train palm values: `557,293 x 6,623 = 3,690,951,539`

### 2.3 全量与抽样统计的区别

以下结果来自完整 557,293 帧：

- 主报告中的 MAE、RMSE、PCC、Temporal Accuracy、Contact IoU、V-IoU Frame-Macro、Pred/GT Volume
- `pressure_bins.csv`
- `false_high_pressure_summary.csv`
- `pointwise_pressure_tails.csv`

以下结果来自 `diagnostic_max_frames=200000` 的随机诊断样本：

- `frame_metrics_sample.csv`
- `frame_bins_by_gt_volume.csv`
- 分 OpenTouch/TouchAnything 的 frame-level 重算
- sampled base-only 与 fused 对照
- sampled catastrophic frame diagnostics

因此，分域和 frame-bin 数值用于定位问题，不应冒充完整训练集精确总体值。

## 3. 一句话结论

模型已经很好地拟合了训练集的**整帧 pressure scale**，但没有精确拟合每帧的**dense vertex topology 与 high-pressure magnitude**。

更准确地说：

- 模型能够判断一帧大致有多少总 pressure。
- 模型能够在大多数训练帧上判断主要接触区域。
- 模型仍会把高压 core 压低，并把一部分 pressure 铺到周围低压或错误 vertex。
- 这种 pressure redistribution 可以保持 total volume，却会明显降低 V-IoU。
- 因此它不是简单的整体欠拟合，也不是只靠增加 epoch 就自然消失的问题。

## 4. 完整训练集总体结果

| Metric | Train-fit |
|---|---:|
| MAE | 0.0085 |
| RMSE | 0.0312 |
| PCC | 0.7937 |
| Temporal Accuracy, threshold 0.05, any-point rule | 0.9439 |
| Contact IoU Frame-Macro, threshold 0.05 | 0.5058 |
| V-IoU Frame-Macro | 0.4824 |
| V-IoU Split-Micro, reconstructed from all pressure bins | 0.5968 |
| Pred/GT Volume | 0.9755 |
| Active MAE, GT > 0.05 | 0.0649 |
| Background MAE, GT <= 0.02 | 0.0031 |
| Active Recall, threshold 0.05 | 0.7996 |
| Background False Positive, threshold 0.05 | 0.0147 |
| Catastrophic Over | 0 / 50,701 |
| Catastrophic Under | 273 / 133,762 = 0.2041% |

### 4.1 这些数字说明什么

`Pred/GT Volume=0.9755` 表明全训练集的总预测 pressure 只比 GT 少约 2.45%。从 total mass 看，模型已经非常接近训练分布。

但 `V-IoU Frame-Macro=0.4824` 表明 pressure 并没有逐 vertex 精确落在 GT 上。总量正确与空间重合正确是两件不同的事。

一个极端例子：

```text
GT   = [1, 0]
Pred = [0, 1]
```

两者 total volume 都是 1，但 V-IoU 为 0。

对非负 pressure，以下恒等式成立：

```text
sum(min(pred, gt)) = (sum(pred) + sum(gt) - L1) / 2
sum(max(pred, gt)) = (sum(pred) + sum(gt) + L1) / 2
```

因此，即使 `sum(pred)` 很接近 `sum(gt)`，只要 pointwise L1 仍较大，V-IoU 就不会高。

## 5. V-IoU 两种口径为什么差这么多

### 5.1 Frame-Macro

```text
V-IoU Frame-Macro = mean_frame(
    sum_vertex(min(pred, gt)) / sum_vertex(max(pred, gt))
)
```

每帧权重相同。GT volume 接近 0 的帧与 GT volume 500 的帧各占一个样本权重。因此它对 empty/low-volume false pressure 非常敏感。

### 5.2 Split-Micro

```text
V-IoU Split-Micro =
    sum_all_frames_vertices(min(pred, gt))
    / sum_all_frames_vertices(max(pred, gt))
```

高-pressure mass 自然贡献更大。使用完整 `pressure_bins.csv` 的 count、mean GT、mean prediction 和 MAE 重建得到：

```text
V-IoU Split-Micro = 0.5968
```

这解释了为什么 0.4824 看起来偏低。它不是 GT 上限，而是 frame-macro 对大量低-volume 帧给予了较高权重。

### 5.3 TouchAnything-style trajectory metric

新评估代码中的 TouchAnything-style 口径为：

```text
每条 source trajectory 内 micro
-> 对 trajectory 等权 macro
```

旧 train-fit 报告没有保存完整 557,293 帧的 per-trajectory sufficient statistics，`frame_metrics_sample.csv` 又只保留 200,000 帧，因此不能从旧结果精确补算该值。需要用更新后的 evaluator 重跑 train split。

## 6. Pressure-bin 校准

| GT bin | Count | Mean GT | Mean Pred | Bias | MAE |
|---|---:|---:|---:|---:|---:|
| `[0, .005)` | 3,051,370,435 | 0.00017 | 0.00246 | +0.00228 | 0.00233 |
| `[.005, .01)` | 99,362,885 | 0.00729 | 0.01753 | +0.01024 | 0.01260 |
| `[.01, .02)` | 112,073,338 | 0.01448 | 0.02527 | +0.01079 | 0.01609 |
| `[.02, .05)` | 147,556,166 | 0.03264 | 0.03940 | +0.00676 | 0.02098 |
| `[.05, .10)` | 103,156,248 | 0.07168 | 0.06607 | -0.00561 | 0.03060 |
| `[.10, .20)` | 89,152,193 | 0.14240 | 0.11167 | -0.03073 | 0.05205 |
| `[.20, .30)` | 38,705,528 | 0.24368 | 0.17577 | -0.06791 | 0.08799 |
| `[.30, .50)` | 29,062,862 | 0.38112 | 0.27460 | -0.10653 | 0.13035 |
| `[.50, .70)` | 10,251,982 | 0.58756 | 0.45524 | -0.13232 | 0.16210 |
| `[.70, 1]` | 10,259,902 | 0.85675 | 0.72617 | -0.13058 | 0.15102 |

### 6.1 分布极度不平衡

| Region | Fraction of all palm values |
|---|---:|
| GT `<0.005` | 82.67% |
| GT `<0.05` | 92.40% |
| GT `>=0.05` | 7.60% |
| GT `>=0.2` | 2.39% |
| GT `>=0.3` | 1.34% |
| GT `>=0.5` | 0.56% |
| GT `>=0.7` | 0.28% |

训练目标绝大部分由 near-zero vertices 构成。即使已有 active pressure weighting，global optimization 仍然容易找到一种折中解：

1. 在 near-zero 区域保留较小正值。
2. 对 high-pressure core 做 shrinkage。
3. 通过扩大中低压覆盖区域维持 frame total volume。

这正是当前 pressure-bin 曲线显示的模式。

### 6.2 高压并非完全没有学会

`GT>=0.7` 时 mean prediction 已达到 0.726，而不是塌缩到 0.1 或 0.2。说明模型具备识别强接触的能力。

但从 `GT>=0.2` 开始出现稳定负偏差，且到 0.3 以上约欠预测 0.11 至 0.13。这说明连续 magnitude 的回归仍存在显著 shrinkage。

## 7. 按 frame GT volume 分层

以下来自 200,000 帧随机诊断样本。

| GT volume range | Frames | Mean GT volume | Mean Pred volume | Median Pred/GT | Mean V-IoU | Mean Contact IoU |
|---|---:|---:|---:|---:|---:|---:|
| exact empty | 591 | 0.00 | 34.54 | N/A | 0.000 | 0.398 |
| about 0 to 11 | 19,941 | 5.88 | 21.93 | 2.98 | 0.214 | 0.256 |
| 11 to 23 | 19,941 | 17.32 | 34.32 | 1.67 | 0.336 | 0.247 |
| 23 to 36 | 19,941 | 29.26 | 42.77 | 1.28 | 0.411 | 0.372 |
| 36 to 52 | 19,941 | 43.44 | 53.09 | 1.11 | 0.447 | 0.420 |
| 52 to 72 | 19,940 | 61.38 | 66.52 | 1.02 | 0.477 | 0.469 |
| 72 to 97 | 19,941 | 83.94 | 82.41 | 0.95 | 0.505 | 0.530 |
| 97 to 127 | 19,941 | 111.37 | 102.89 | 0.91 | 0.542 | 0.592 |
| 127 to 168 | 19,941 | 146.65 | 129.20 | 0.88 | 0.577 | 0.651 |
| 168 to 243 | 19,941 | 200.56 | 172.26 | 0.86 | 0.613 | 0.715 |
| 243+ | 19,941 | 436.24 | 402.50 | 0.94 | 0.715 | 0.808 |

### 7.1 清晰的体积分层现象

- Empty/near-empty frames 被显著高估。
- 低 volume 区间从 3 倍、1.7 倍逐渐回落。
- 中等 volume 附近最接近 1。
- 较高 volume 开始系统性欠预测。
- 最大 volume bin 的比例反而恢复到约 0.94，说明极强整帧接触相对容易识别。

这不是简单的全局 scale 偏大或偏小，而是一条明显的回归到中间值曲线。

## 8. 分域训练集拟合

以下来自 200,000 帧诊断样本，而不是完整分域 train eval。

| Dataset | Frames | Frame MAE | V-IoU Frame-Macro | Contact IoU | Pred/GT Volume | Frame-volume Corr | Cat Under |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenTouch | 45,800 | 0.00831 | 0.5268 | 0.5606 | 0.9581 | 0.8958 | 0.0454% |
| TouchAnything | 154,200 | 0.00860 | 0.4691 | 0.4891 | 0.9804 | 0.9647 | 0.1943% |
| Mixed | 200,000 | 0.00853 | 0.4823 | 0.5055 | 0.9761 | 0.9610 | 0.1669% |

TouchAnything 的 frame-volume correlation 更高，但 dense IoU 更低。这说明 TA 的整帧 scale 更容易预测，而具体 pressure 放在哪些 vertices、以何种幅值展开更难。

OpenTouch 的 dense overlap 更高，但 frame-volume correlation 略低。两个数据域的难点并不相同，mixed training 不能只看一个平均 RMSE。

## 9. False-high 与背景安全性

### 9.1 完整训练集 pointwise false-high

以 GT `<0.005` 为背景：

| Prediction threshold | False-high rate | Mean pred in false-high | Excess volume / total pred volume |
|---|---:|---:|---:|
| `>=0.05` | 0.9016% | 0.0821 | 3.6686% |
| `>=0.10` | 0.1622% | 0.1501 | 1.2075% |
| `>=0.20` | 0.01951% | 0.3125 | 0.3022% |
| `>=0.30` | 0.00694% | 0.4492 | 0.1547% |
| `>=0.50` | 0.00195% | 0.6440 | 0.0623% |
| `>=0.70` | 0.00056% | 0.7911 | 0.0221% |

训练集上的严重 pointwise false-high 已经很少，但并非为零。`GT<0.005, Pred>=0.3` 仍有 211,899 个 palm values，不过它们只占对应背景值的 0.00694%，excess volume 占全部预测 volume 的 0.1547%。

### 9.2 Catastrophic frame

完整 train report：

- Catastrophic over: 0 / 50,701
- Catastrophic under: 273 / 133,762 = 0.2041%

训练集基本没有 `GT volume <10` 且 `Pred volume >300` 的极端过预测帧。说明 test 上出现的 catastrophic over 主要是泛化问题，而不是模型在训练集上也无法压住该模式。

## 10. Base-only 与 fused prediction

以下来自 200,000 帧 sampled base/fused 对照。

| Diagnostic | Result |
|---|---:|
| Fused/Base mean volume ratio | 1.0234 |
| Base false-high excess, GT<.005 and Pred>=.3 | 40,694.3 |
| Fused false-high excess | 34,717.8 |
| False-high excess reduction | 14.69% |
| Residual-created catastrophic over | 0 |
| Residual-corrected catastrophic over | 2 |

分域结果：

- OpenTouch false-high excess下降约 24.69%。
- TouchAnything false-high excess下降约 14.40%。

因此，ReZero residual 在训练样本上不是单纯制造高压。它略微增加 total volume，同时在 sampled frames 中减少 false-high excess，并纠正了 2 个 base catastrophic frame。

但这不等于 residual 一定改善泛化。训练集上的 correction 与 test sequence 上的 residual co-adaptation 可以同时存在，这正是后续 strict-control 与 BaseAnchor 对照需要验证的内容。

## 11. 训练集与测试集的泛化差距

BaseAnchor RMSE-best：

| Split | MAE | RMSE | PCC | V-IoU Frame-Macro | Pred/GT Volume | Cat Over | Cat Under |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixed train | 0.0085 | 0.0312 | 0.7937 | 0.4824 | 0.9755 | 0.000% | 0.204% |
| OpenTouch test | 0.0103 | 0.0425 | 0.8373 | 0.4708 | 0.8755 | 0.000% | 1.639% |
| TA seen | 0.0114 | 0.0414 | 0.6884 | 0.3613 | 1.0307 | 1.230% | 4.633% |
| TA unseen | 0.0132 | 0.0464 | 0.6753 | 0.3424 | 0.8012 | 0.000% | 9.762% |

### 11.1 主要差距不只是常规 RMSE

Train 到 test 的 RMSE 上升明显，但更重要的是：

- TA seen 出现训练集没有的 catastrophic over。
- TA unseen 的 catastrophic under 接近 10%。
- TA 的 V-IoU 从 sampled train 的约 0.469 降到 0.361/0.342。
- TA unseen Pred/GT Volume 下降到 0.801。

模型在训练集上已经能够拟合 frame scale，因此 test failure 不能全部归因于 head capacity 不够。至少还存在 domain/sequence/query 泛化问题。

## 12. 为什么训练集 V-IoU 没有接近 1

### 12.1 目标函数没有直接优化 V-IoU

当前 V2 loss 由 weighted SmoothL1、BCE-with-logits、background penalty 等组成。它们鼓励 pointwise accuracy，但并不等价于最大化 soft IoU。

当 92.4% palm values 低于 0.05 时，降低大量背景误差通常比精确恢复极少数高压 core 更容易降低训练 loss。

### 12.2 Frozen representation 限制

DINO 完全冻结。Head 只能从 frozen visual tokens 中读取 pressure 相关线索。如果图像对连续 force magnitude 本身不可观测，或 DINO feature 没有保留足够局部信息，head 无法通过继续训练凭空恢复它。

### 12.3 Decoder bottleneck

当前路径最终仍经过：

```text
16x12 features
-> 5x5 pooling, retain 21 cells
-> global 512 representation
-> 13,614 dense outputs
```

从 21 个 spatial cells 与 global 512 latent 恢复 6,623 个 palm pressure values，本身就是很强的信息压缩。模型更容易学到平滑、平均化的 pressure template，而不是每帧精确局部拓扑。

### 12.4 Dropout 与正则化

V2 decoder 包含较强 Dropout。它有助于泛化和稳定，但降低了纯 memorization 能力。当前 train-fit 是正常训练配置的结果，不等价于“关闭所有正则后，这个网络是否能记住训练样本”。

### 12.5 静态 RGB 的可观测性上限

同样的手部外观可能对应不同的真实 force，尤其是：

- 材料刚度不同。
- 接触法向力不同但形变不明显。
- 物体在 crop 外提供约束。
- 当前帧无法区分正在加压、保持还是释放。

因此，单帧 RGB 到连续 force 并不总是一一映射。这个不可观测性上限不是 GT V-IoU 上限，而是当前输入条件下可学习函数的上限。

### 12.6 标签平滑与跨域语义

OpenTouch 与 TouchAnything 的 pressure construction、diffusion、有效区域和时间语义不同。即使 data-integrity audit 没有发现错帧或 target serialization mismatch，也不代表两个域的同一归一化 pressure 数值具有完全相同的物理语义。

## 13. 目前可以确认与不能确认的事情

### 13.1 可以确认

1. GT 的 V-IoU 理论上限不是 0.48。`pred=GT` 时为 1。
2. 模型已经很好拟合 train frame-volume scale。
3. 模型没有精确拟合 train dense pressure field。
4. 训练集误差具有明确的 low-over/high-under shrinkage 模式。
5. 严重 false-high 在训练集很少，test false-high 主要是泛化问题。
6. BaseAnchor residual 在 sampled train frames 上整体减少了 false-high，而不是只制造 upward pressure。

### 13.2 不能确认

1. 不能仅凭本次 train-fit 断言模型结构绝对无法达到更高 V-IoU。
2. 不能把 0.4824 当成数据集 GT 上限。
3. 不能从旧 train-fit 报告精确补算 TouchAnything trajectory-level metrics。
4. 不能仅靠 total volume ratio 判断 dense field 已正确拟合。
5. 不能仅凭 data-integrity audit 排除视觉语义上的脏标签。

## 14. 最关键的后续诊断

### 14.1 Small-subset memorization test

分别固定 256 和 1,024 个样本：

- 关闭 augmentation。
- 关闭 decoder Dropout。
- 使用固定 batch schedule。
- 训练到 loss 明显饱和。
- 同时记录 V-IoU Frame-Macro、Split-Micro、TouchAnything-style trajectory V-IoU。

解释：

- 若 V-IoU 可以达到 0.9 以上，说明模型有表达能力，完整训练集问题主要来自正则、分布不平衡、优化和泛化。
- 若小样本仍停在约 0.6，说明 frozen features、5x5/512 bottleneck 或目标函数确实限制了可拟合性。
- 若关闭 Dropout 后明显上升，说明此前 train-fit 不能作为 architecture capacity 上限。

### 14.2 使用新 evaluator 重跑 train-fit

更新后的 evaluator 会额外输出：

```text
V-IoU Frame-Macro
V-IoU TA-Trajectory
V-IoU Split-Micro
TA Temporal Accuracy
TA Temporal F1
TA Contact IoU
TA MAE
```

并保存 `touchanything_protocol_sequence_metrics.csv`。这样可以直接找出训练集中是否存在少数始终无法拟合的 trajectory。

## 15. 最终判断

当前 BaseAnchor 并不是“连训练集都不会”。它已经学会了相当强的 frame-level pressure scale，并且训练集极端错误很少。

真正的问题是：

```text
frame scale 拟合较好
!= dense tactile topology 拟合充分
!= continuous high-pressure magnitude 拟合充分
!= test sequence 泛化安全
```

因此，下一步不应只增加 epochs 或把学习率继续调大。应先用 small-subset memorization 区分表达能力上限与完整分布优化问题，再结合 strict-control、stratified sequence sampling 和新 TouchAnything-compatible metrics 判断泛化误差来自哪里。

## 16. 源文件

- Full train report: `hamer_tactile_ft/eval_reports_mixed_dense_v2_dinov3_rezero_baseanchor_trainfit/rmse-best/mixed_train/eval_opentouch_touchanything_train.txt`
- Pressure bins: `.../eval_opentouch_touchanything_train_diagnostics/pressure_bins.csv`
- Frame bins: `.../eval_opentouch_touchanything_train_diagnostics/frame_bins_by_gt_volume.csv`
- Frame sample: `.../eval_opentouch_touchanything_train_diagnostics/frame_metrics_sample.csv`
- False-high summary: `.../eval_opentouch_touchanything_train_diagnostics/false_high_pressure_summary.csv`
- Pointwise tails: `.../eval_opentouch_touchanything_train_diagnostics/pointwise_pressure_tails.csv`
- BaseAnchor test reports: `hamer_tactile_ft/reports/eval_reports_mixed_dense_v2_dinov3_rezero_baseanchor/`
