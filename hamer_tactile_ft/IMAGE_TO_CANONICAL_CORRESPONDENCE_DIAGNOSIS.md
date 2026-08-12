# 图像 Query 到 Canonical Mesh 对应问题：六项诊断总结

## 文档目的

本文总结当前 TouchAnything 单域主线中六个相互关联的问题：

1. Depth 局部融合没有超过 FullGrid baseline。
2. Cross-attention/CSE 没有超过 MLP/FullGrid。
3. Canonical mesh 上的接触位置仍不准确。
4. VLM 全局语义 probe 较弱。
5. 高压区域存在系统性幅值不足。
6. False-high 高度集中于特定动作和序列。

核心判断是：当前 SAM3 bbox/mask 已经较好地定义了匿名目标 query，但它们没有提供
图像坐标系到 canonical hand mesh 的稳定对应。这个缺失会直接伤害 Depth、attention
和位置预测，也会间接造成 pressure 扩散、高压压缩和动作 shortcut。不过，它不能
单独解释所有问题；单帧不可观测性、标签分布和训练目标仍然是独立因素。

本文只总结诊断和后续边界，不把尚未验证的假设写成既定结论。

> 代码状态：VLM V1-V6 probe 和首轮正式 Depth adapter 已完成并从活动源码中移除；
> 结果与研究结论没有删除。历史结论见
> [Input-Prior Research History](../tactile_input_priors/HISTORY.md)，当前保留的可复用
> 实现仅为离线 MoGe depth sidecar 基础设施。

## 三个容易混淆的层次

### 1. Query localization

BBox 定义当前需要预测的匿名目标：

```text
完整图像中的哪一个局部区域，是当前 tactile query？
```

SAM3 bbox质量提升主要解决这一层。Crop scale决定保留多少目标和上下文，但bbox本身
不告诉模型目标表面各位置的身份。

### 2. Foreground segmentation

SAM3 mask提供：

```text
哪些像素属于目标手，哪些属于背景或物体？
```

它可以减少背景污染和无关token，却仍然只给出二值前景，不包含拇指、食指、掌心或
canonical vertex编号。

### 3. Dense correspondence / registration

触觉输出真正需要的是：

```text
当前图像中的pixel/patch，对应canonical mesh上的哪些vertices？
```

这个对应会随视角、手部关节、遮挡、左右镜像和物体遮挡发生变化。完美bbox和mask
也不能自动解决它。当前系统的困难更准确地说是：已经有query localization，但缺少
query registration。

## 现有实验形成的证据链

- 严格数据完整性审计没有发现足以解释整体结果的错帧、损坏或target mismatch。
- SAM3 bbox改善了query定位，但没有让空间先验分支稳定超过FullGrid。
- FullGrid在1024样本memorization中达到约`.953/.944`的Frame-Macro V-IoU，说明
  frozen DINO + dense decoder具有足够的样本内表达能力。
- 完整训练中frame pressure scale可拟合得较好，但canonical位置明显弱于训练集幅值，
  表明容量并不是唯一瓶颈。
- Full-vertex cross-attention、deformable attention和CSE-style映射都没有提供稳定的
  spatial-control优势。
- 最新Depth Local Residual中，Spatial略优于Global Control，但打乱depth空间位置后
  unseen反而改善，否定了“模型已经学到可泛化depth-to-mesh对应”的解释。

相关结果：

- [Depth Spatial报告](eval_reports_ta_dlocal_sp_r256/)
- [Depth Global报告](eval_reports_ta_dlocal_glb_r256/)
- [Depth Spatial Shuffle报告](eval_reports_ta_dlocal_sp_shuffle/)
- [VLM与Depth历史结论](../tactile_input_priors/HISTORY.md)

## 问题一：Depth 局部融合失败

### 观察到的结果

在相同frozen FullGrid base上，正式`loss-best`结果为：

| 模型 | Seen TA Contact | Unseen TA Contact | Seen TA V-IoU | Unseen TA V-IoU | Seen/Unseen CoreLoc |
|---|---:|---:|---:|---:|---:|
| Frozen FullGrid base | **.5402** | **.4618** | **.5142** | **.4442** | `.4077/.3774` |
| Global Depth | .5371 | .4509 | .5108 | .4369 | `.4078/.3760` |
| Spatial Depth | .5386 | .4541 | .5141 | .4385 | `.4083/.3768` |
| Spatial checkpoint + shuffle | .5377 | .4587 | .5126 | .4434 | `.4072/.3770` |

Spatial相对Global平均只提高约`.00236` Contact、`.00248` V-IoU和`.00071`
CoreLoc，仍然没有超过base。更关键的是，同一Spatial checkpoint打乱depth空间位置后，
unseen Contact和V-IoU分别再提高约`.00463/.00493`。

### 为什么缺少registration会导致这个结果

Depth teacher输出的是camera/image坐标系中的相对几何：

```text
哪些pixel更近、哪里有深度边界、表面朝向如何
```

而最终监督位于canonical mesh：

```text
哪些canonical vertices接触、每个vertex压力是多少
```

Depth residual因此被迫在没有显式监督的情况下同时解决：

1. 目标手与物体分离。
2. 手部区域/手指身份识别。
3. 2D image到canonical surface registration。
4. 手物几何是否支持接触。
5. 对6,623个有效vertices进行pressure修正。

这比“使用depth判断局部接近关系”重得多。模型更容易利用depth分布的全局统计调节
整帧pressure volume，而不是学习稳定的局部几何对应。

### 它实际上学到了什么

`loss-best`中的up/down correction volume为：

| Split | Global up/down | Spatial up/down | Shuffle up/down |
|---|---:|---:|---:|
| Seen | `1.18/5.64` | `2.16/4.69` | `3.02/2.90` |
| Unseen | `1.08/6.18` | `2.14/4.67` | `2.98/2.99` |

Global主要成为全局下调器；Spatial少下调一些；Shuffle几乎恢复上下对称。对应的
Pred/GT volume也逐渐恢复到base。这说明主要收益来自calibration，而不是位置。

### 训练动态

Spatial的train full-ramp loss从`.08047`降至`.07747`，但val full-ramp loss从
`.09580`升至`.09680`。其loss-best已经在epoch 0；到last时residual RMS从约
`.066`增至`.341`，饱和率达到`14%-15%`，Contact和CoreLoc明显退化。

因此，bounded residual防住了数值爆炸，却没有防止分支记忆训练集并逐渐吞噬base。
继续放宽bound、扩大网络或延长训练都不对症。

### 当前结论

- 不能得出“depth完全无信息”。Spatial略优于Global说明存在弱信号。
- 可以得出“当前自由depth-to-vertex residual没有学到可泛化局部对应”。
- 若继续depth，应先在image/token空间预测contact plausibility、遮挡或距离风险，或
  只进行受约束的单向安全抑制，不应继续自由修正全部vertices。

## 问题二：Cross-Attention/CSE 没有超过 MLP

### 为什么attention不会自动建立对应

Cross-attention提供了一个灵活的读取机制：

```text
vertex query从image tokens中选择信息
```

但它没有告诉模型某个query应当读取哪个token。没有dense correspondence、part label、
canonical anchor或pose监督时，许多attention排列都能获得相近训练loss。模型可以选择
更容易的解：

- 所有vertices读取相似的全局token统计。
- learned vertex embedding记忆canonical pressure先验。
- attention只承担额外容量，真正预测仍来自base。
- seen数据上记忆动作/视角到mesh模板，但不能迁移到unseen。

因此，“更现代”或“层数更多”并不保证优于MLP。FullGrid MLP虽然没有显式attention，
但flatten后的固定token顺序本身就提供了稳定的位置索引；在当前监督下，它反而更容易
优化。

### CSE-style为什么也困难

CSE或DensePose式方法通常需要某种显式surface correspondence监督，例如：

- image pixel对应的canonical UV/XYZ。
- body part或surface anchor类别。
- 多视角/渲染产生的对应。
- 已注册mesh的投影。

当前pressure target只有canonical vertex值，没有说明图像中的哪块区域对应哪个vertex。
因此CSE分支仍可能退化为learned mesh template或global calibration。

### 现有control的含义

当Spatial与Global Control差距很小，或token/depth shuffle没有稳定破坏结果时，应解释为：

```text
新增参数或全局统计可能有作用，但没有证明动态空间对应有效
```

这也是此前XAttn、CSE和最新Depth Spatial结果共同指向的结论。

### 当前结论

- 不再通过增加heads、层数、query维度或放宽gate来补救。
- 若未来恢复attention路线，前置条件应是先获得可验证的part/anchor/canonical坐标监督。
- Attention更适合作为已有correspondence上的局部读取器，而不是在只有pressure回归监督时
  独自发现完整registration。

## 问题三：Canonical Mesh 接触位置不准确

### 第一性原理

当前模型学习：

```text
RGB crop -> 13,614维canonical pressure
```

但影响图像外观的隐变量包括：

```text
camera viewpoint / hand articulation / handedness / occlusion /
object geometry / contact state / applied force
```

如果没有坐标桥梁，模型只能从数据共现中隐式消除这些变化。训练集上可以记忆，跨动作、
跨视角和unseen物体时则容易输出平均模板。

### 位置错误如何污染其他指标

假设GT是：

```text
10个正确vertices x 0.8 = volume 8
```

位置不确定的回归器可能输出：

```text
40个附近vertices x 0.2 = volume 8
```

整帧volume正确，但会同时出现：

- peak pressure不足。
- Contact区域扩散。
- CoreLoc和Distribution V-IoU下降。
- 低压halo增加。
- GT高压bin的mean prediction偏低。

这就是“幅值拟合尚可但位置差”的一种来源。Smooth L1/MSE类回归在多种可能接触位置
之间倾向预测条件均值，从而进一步产生扩散。

### 为什么不再主要归因于decoder容量

FullGrid在小样本memorization中可以高度拟合，说明模型能够表达尖锐的canonical
pressure分布。完整数据泛化失败更可能来自：

- 输入到canonical坐标的映射不稳定。
- 相似RGB对应多个合法pressure target。
- 遮挡使真实接触位置不可见。
- 训练分布鼓励动作/物体模板shortcut。

### 推荐的定位诊断

后续位置分析应至少拆分：

1. `distribution_viou`：消除总volume尺度影响。
2. `core_distribution_viou`：强调真实高压core而非低压halo。
3. GT support上的predicted mass和prediction support中的GT mass。
4. Predicted peak到GT high-pressure区域的mesh geodesic distance。
5. Oracle-volume-scaled prediction：判断错误来自总量还是位置。
6. GT-support oracle magnitude：判断正确区域内是否仍系统性低估。

### 当前结论

位置应继续作为第一优先级，但重点应从“扩大decoder”转到“建立、监督和审计
image-to-canonical correspondence”。

## 问题四：VLM 全局语义 Probe 较弱

### VLM能提供什么

VLM较擅长识别：

```text
物体类别 / interaction action / grasp vs pinch vs support /
材料和可压缩性 / 是否可能存在接触
```

这些是global context，不天然包含精细的canonical位置或真实法向力。

### 为什么只输入crop可能削弱VLM

- 物体可能只露出局部纹理，无法识别完整物体。
- 动作上下文被裁掉。
- 无法判断当前手与哪个物体发生交互。
- 紧crop更接近局部视觉encoder的输入，不容易体现VLM的语义优势。

但直接输入整图也会引入多只手、多人和多个物体。更合理的语义输入是：

```text
完整场景 + 显式标记的匿名query bbox + query crop
```

标记只用于说明当前query，不需要向模型输入人物、左右手或dataset身份。

### 为什么registration不是VLM弱的唯一原因

“拿杯子”可以对应五指包覆、捏住杯沿、托住杯底或掌心悬空。它们在VLM语义空间
中可能接近，但pressure分布完全不同。即使VLM正确识别动作，它也无法从语义唯一恢复
contact位置和magnitude。

### 合理的VLM职责

VLM更适合预测或调制低维状态：

- non-contact/contact概率。
- grasp/pinch/support/press等粗动作。
- 接触面积等级。
- finger-dominant或palm-likely。
- 当前视觉是否存在高歧义。

这些条件可以用于scale-only FiLM、风险判断或受限gate，但不应无界地直接制造vertex
pressure。必须保留DINO-wide、constant、within-sequence shuffle等control，避免把额外
视觉容量误称为语言语义收益。

### 当前结论

- VLM probe弱不能简单归因于bbox/mask。
- Crop缺少全局上下文是一个可能原因，但语义到细粒度触觉本身就是多对多映射。
- VLM仍可能改善frame-level contact state或ambiguity，不适合作为canonical定位主干。

## 问题五：高压幅值系统性不足

### 四类可能来源

高压欠预测至少可以分为：

1. **位置摊薄**：不知道正确位置，把少量高压扩散成大面积中低压。
2. **标签边际与loss**：高压vertices稀少，回归目标倾向conditional mean。
3. **单帧不可观测性**：RGB和单目depth不能直接观测法向力、材料刚度和微小形变。
4. **模型/输出压缩**：activation、weight形状和正则进一步抑制极值。

### 如何区分位置和幅值

#### GT-support oracle

只在GT真实接触或高压区域检查prediction：

```text
如果正确区域内仍然很低 -> magnitude监督或可观测性问题
如果正确区域尚可但峰值出现在别处 -> correspondence问题
```

#### Oracle volume scaling

把prediction按比例缩放到GT frame volume，再看Distribution/CoreLoc：

```text
缩放后仍然很差 -> 主要是位置问题
缩放后显著恢复 -> 主要是frame magnitude问题
```

#### Peak与geodesic诊断

同时记录peak ratio和predicted peak到GT core的mesh geodesic distance，可以区分：

- 峰值正确但整体偏低。
- 峰值强度合理但位置错误。
- 峰值既低又远。

### 为什么不能只继续提高高压weight

如果模型位置不准，提高高压权重可能把错误位置上的`.2`抬成`.5`，直接增加最危险的
false-high。此前Tail L1和多种magnitude设计也说明，单纯强化稀有高压监督容易牺牲
低压安全和位置。

### Depth和VLM的能力边界

- Depth最多提供接近、遮挡和表面方向，不能直接提供真实force。
- VLM最多提供物体、动作和材料先验，不能知道当前帧实际施力。
- 两者可以帮助判断“高压是否合理”，但不能替代magnitude监督和真实受力证据。

### 当前结论

高压问题应在位置诊断之后处理。只有确认GT support上的幅值仍不足，才值得比较
capped monotonic weight、Balanced Regression、BerHu或独立frame magnitude head。

## 问题六：False-High 集中于特定动作和序列

### Shortcut形成机制

TouchAnything训练数据可能包含如下强共现：

```text
看到grasp/holding外观 -> 多根手指和掌心通常都有pressure
```

模型于是学到一个宽而稳定的动作模板。当测试帧是捏取、悬空握持、释放阶段或只有指尖
接触时，掌心和无关手指仍会被抬高。

错误集中特定序列也可能来自：

- 同一长序列的高度重复样本强化单一pressure模板。
- 物体遮挡了真正接触区域。
- Crop中存在共同可见的其他手或物体区域。
- Query bbox短缺口、跨实例切换或FOV clipping。
- 相似动作在训练集中缺少低接触/无掌心接触的hard negatives。

### Registration如何参与

模型可能已经正确判断“发生了接触”，却不知道接触位于指尖还是掌心。缺少canonical
对应时，最安全的训练集平均解就是把pressure分散到常见掌部区域。因此动作识别正确
并不代表vertex位置正确。

### 为什么VLM可能放大问题

如果VLM识别出“grasp”，然后通过无约束FiLM整体增强feature，可能形成：

```text
grasp语义 -> 掌心和多根手指一起升压
```

这会加强而不是消除动作shortcut。VLM应更多用于风险或粗接触状态，不能单独触发新的
高压区域。

### 为什么单目Depth也不一定解决

真实手物接触经常表现为：

- 手与物体深度连续，边界不明显。
- 接触区域被物体遮挡。
- 透明、反光或纹理弱物体产生depth误差。
- 相隔几毫米与真正接触在单目relative depth中难以区分。

因此depth适合判断局部几何是否支持已有RGB候选，不能被当作contact真值。

### 更可靠的数据与模型方向

最有价值的是同动作hard negatives，例如：

```text
同样拿杯子：掌心接触 / 仅指尖接触 / 看似握持但实际悬空
```

这能迫使模型放弃“动作名称到固定pressure模板”。模型侧可采用asymmetric condition：

- RGB base产生候选pressure。
- VLM判断动作、接触状态和歧义。
- Depth判断局部几何是否支持接触。
- 辅助分支可以较强地下调不可信候选。
- 只有RGB局部证据和辅助先验共同支持时，才允许小幅上调。

### 当前结论

False-high是数据shortcut、query质量、遮挡、缺少registration和单帧歧义共同造成的，
不能依赖单个更强backbone或更高pressure weight解决。

## 六项问题的关系矩阵

| 问题 | Registration相关性 | 主要独立因素 |
|---|---|---|
| Depth局部融合失败 | 高 | teacher噪声、relative depth能力边界、adapter过拟合 |
| Cross-attention/CSE失败 | 高 | 缺少对应监督、优化退化、global/template shortcut |
| Canonical位置不准 | 核心 | 遮挡、单帧多解、标签扩散 |
| VLM语义probe弱 | 中低 | crop上下文不足、语义与pressure多对多、VLM目标不匹配 |
| 高压幅值不足 | 中 | 高压稀有、loss conditional mean、真实force不可观测 |
| 特定动作false-high | 高 | 动作共现、长序列重复、query污染、hard negative不足 |

## 第一性原理下的系统简化

当前系统不应继续要求每一种新先验都直接输出canonical vertex residual。更清晰的职责
划分是：

```text
SAM3 bbox/mask
    -> 定义匿名query和目标前景

RGB/DINO
    -> 主要的局部视觉与pressure base

Correspondence/part anchors
    -> 建立image tokens到canonical surface的坐标桥梁

Depth
    -> 局部几何、遮挡和contact plausibility

VLM
    -> 物体、动作、粗接触状态和歧义

受约束融合
    -> 优先抑制不可信接触，谨慎上调有共同证据的候选
```

这比让VLM或Depth自由修正6,623个有效vertices更符合它们各自提供的信息。

## 建议的后续优先级

1. 保持当前FullGrid作为正式baseline，不合并失败的Depth Local Residual。
2. 增加part/anchor/canonical coordinate probe，验证是否能建立可泛化的
   image-to-canonical对应。
3. 增加GT-support、oracle-volume和mesh geodesic诊断，把位置和幅值错误彻底拆开。
4. VLM改为query-aware full-scene contact-state/ambiguity probe，不直接回归mesh pressure。
5. Depth改为image/token-space contact plausibility或one-sided suppression，并保留RGB edge、
   global和shuffle control。
6. 构造同动作不同接触状态的hard negatives，优先处理掌心误高等危险shortcut。
7. 只有位置稳定后，再独立测试高压weight、magnitude head或balanced regression。

## 当前不应做的事情

- 不继续放宽Depth residual或ReZero gate/budget。
- 不通过堆叠attention层、heads或query维度期待自动产生registration。
- 不把SAM3 mask等同于canonical correspondence。
- 不让VLM的grasp语义直接无界抬升pressure。
- 不在位置尚不可靠时继续激进提高高压weight。
- 不因为当前融合失败就直接得出“VLM或Depth完全无用”。

## 最终总结

六项问题的共同核心不是单纯的bbox质量或decoder容量，而是当前系统缺少一个经过监督和
审计的image-to-canonical coordinate bridge。这个缺失对Depth、attention和位置预测的
影响最大；对VLM、高压幅值和动作false-high则是部分原因。

因此后续应把问题拆成三个层次：

```text
1. 当前query是谁、前景在哪里？
2. 图像证据对应canonical surface哪里？
3. 对应位置上的contact和pressure magnitude是多少？
```

SAM3主要解决第一层；当前最大的结构缺口在第二层；VLM和Depth应作为第二、第三层的
受限辅助证据，而不是绕过对应关系直接生成完整canonical pressure。
