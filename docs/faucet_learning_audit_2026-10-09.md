# Faucet learning audit — 2026-10-09

本次只检查本地代码、54 条数据、保存的 checkpoint 元数据和日志，并查询公开一手报告；没有启动云端训练或 rollout。没有修改现有实验的训练配置或 normalization assets。

## 能确认的事实

- 数据：54 条 H50 TurnOnSinkFaucet human demos，12070 帧，20 Hz，总计约 10.06 分钟。
- 初始化：本次配置来自官方 pi05_base；不能将它描述成从零训练的模型，它是未经本任务微调的预训练模型。
- batch size 8；Transformer LoRA 共 49,987,584 个参数。Vision、action_in_proj、action_out_proj、time_mlp_in/out 均冻结。
- loss 窗口平均：1–100 步 0.391074；501–700 步 0.128494；1001–1200 步 0.111323；2801–3000 步 0.109636；4801–5000 步 0.103038。说明初期有学习，随后改善很慢。
- 初始 peak LR 5e-6，3000 步衰减到 5e-7；3000→5000 和 5000→7000 续训保留原 schedule，在 5e-7 尾部运行。
- 3000/5000/7000 rollout 分别 0/20、1/20、0/20。旧 runs 初始场景并不严格相同，不能把差异当作配对评估或过拟合证据。
- 上游 loss 是 flow-matching velocity MSE，先在 32 个动作维度上平均，再跨 batch/horizon 平均；无动作维度 loss mask。原始 12D action 只有前 6 维变化，其余 6 维恒定，另补 20 维零。常量目标也有加噪过程，不能假设其 loss 恰好为零，更不能把总 loss 0.1 直接换算为动作 RMSE。

## 优先级最高：状态离散化的信息损失

当前 pipeline 对 state 使用 z-score，但 pi05 离散状态编码使用 np.digitize(state, np.linspace(-1,1,257)[:-1])-1。小于 -1 的值都成为 -1，达到最后一个边界约 0.9922 的值都成为 255；不同位置会合并为相同编码。

对全部 12070 帧重算，结果与 checkpoint 中 state std 一致：

| 末端位置维度 | 当前两端饱和比例 | 仅在 CPU 上试算 q01/q99 映射后的比例 |
|---|---:|---:|
| x | 54.67% | 2.21% |
| y | 48.14% | 2.10% |
| z | 42.46% | 2.17% |

这是已确认的信息损失。它是否是成功率低的主要原因还需要对照训练；视觉也提供部分位置信息，不能凭此断言修好就一定成功。

候选修复是仅调整 state 的归一化范围，并保持动作归一化可控；不要只给现有 checkpoint 换一份 stats 做 rollout。新的输入编码需要独立实验，从相同 pi05_base 开始，训练与评估使用同一份新 stats。动作与状态不必强制共用同一种归一化。

## 其他已确认的设置与待验证风险

1. 自定义冻结范围比 pinned OpenPI 的 get_freeze_filter() 更严格。上游默认 LoRA filter 只冻结 llm 非 LoRA 参数，不自动冻结 vision 或 action/time heads；本项目手动冻结所有非 LoRA 参数。这不必然错误，但新机器人接口的适配能力可能受限。
2. 可先保持 vision 冻结，单独允许 action_in_proj/action_out_proj 和 time_mlp_in/out 训练：按 checkpoint shape 合计增加 2,165,792 参数。不是全量微调，也不能仅凭参数量保证显存一定够；需要短跑实测。
3. 续训 LR 已很低。它可能解释改善缓慢，但没有测每层梯度和实际 update norm，不能证明低 LR 是唯一原因。现有脚本取到了 upstream grad_norm，但写 metrics 时没有保存它。
4. 训练视频原始 128×128 被放大到 256×256，rollout 是真实 256×256 render，之后模型再 resize 到 224。训练/推理图像频谱和细节不完全一致；需要检查同场景经过相同成像处理后的输入。
5. 50-step chunk 相当于 2.5 秒预测，rollout 每次执行前 5 步。全 chunk 平均 loss 不能告诉我们实际执行的前 5 步是否准确。应拆开前 1/5/50 步、前 6 动作维度和常量/补零维度报告。
6. 模态文件把全部 action 标为 absolute，Groot 读取器因此在 episode 尾部重复最后 action；所有可抽样 chunk 的约 10.96% target 位置落在 episode 结尾之后。54 条里只有 5 条最后的 arm action 非零，因此暂时不是首要解释，但应核对原始控制语义、尾部 padding 和 loss mask。
7. 尚未做同场景 GT action replay，尚未证明能在 1–2 条训练 demo 上稳定重现动作，尚未测 held-out demonstration action error。因此不能说“训练集已拟合，只有泛化不好”，也不能保证 54 条足够覆盖所有厨房/水龙头条件。

## 公开的一手报告

- OpenPI issue 1060（2026-10-04）：作者报告 100 条 scripted demonstrations 的仿真任务中，LoRA 两次运行均 0/150，全量 60/150，而末期 loss 和 held-out open-loop action error 接近。LR 与 EMA 设置并非完全相同，也没有维护者给出的已确认根因，不能视为 LoRA 必然失败的证明。
  https://github.com/Physical-Intelligence/openpi/issues/1060
- Dream Machines 的实际双臂制造任务：全量 76%；LoRA rank64 两个配置 45% / 32.5%，rank256 52.5%。数据规模、机器人、实现和训练预算与本实验不同。它证明存在明显差距的实际案例，同时 LoRA 也能取得非零且有用的成功率。
  https://dream-machines.eu/blog/pi05-fine-tuning
- OpenPI issue 906：作者报告 Franka Panda 120 条训练 demos、30k steps，train loss 0.0312 / val loss 0.0270，仍无法可靠完成最初抓取。未给出确认修复。
  https://github.com/Physical-Intelligence/openpi/issues/906

不要把其他任务的 0.002 / 0.03 当成本任务必须达到的 loss 阈值；归一化、动作维度、horizon 和任务分布不同。

## 下一次最小诊断顺序

1. 对可恢复原始场景/状态的 demonstration 做 GT action replay；如果缺原始 simulator 状态，先明确缺项，不能用随机同名 task 替代并宣称 replay 通过。
2. 可先离线测现有 5000 checkpoint 在训练 demo 观测上的动作误差，固定噪声，报告真实 6D arm error 和前 1/5/50 步。区分根本没拟合与离开专家轨迹后失稳。
3. 准备 1–2 条 demonstration 的固定样本拟合检查；先对照修正 state normalization，再逐项试小 heads 解冻与学习率。固定样本/噪声指标只用于检查拟合能力，另外保留新噪声和闭环验证，避免只记住噪声。
4. 通过小样本拟合和同场景闭环后再训练全部 54 条；预留少量完整 episodes 做验证。记录 grad/update norm、分动作维度误差、接近把手/接触/转动进度。
5. 用同一保存的 XML、完整 simulator/model 状态及图像核验条件做 checkpoint 对照；新的 scene replay 实现仍未完成全 20 条验证。

可复算脚本：audit_faucet_learning_local.py；数值结果：learning_audit_local.json。

源码依据：
- pinned loss: https://github.com/robocasa-benchmark/openpi/blob/5a6beda9ff99da30b4e1b59320f6a32971d7c397/src/openpi/models/pi0.py
- pinned freeze filter: https://github.com/robocasa-benchmark/openpi/blob/5a6beda9ff99da30b4e1b59320f6a32971d7c397/src/openpi/models/pi0_config.py
- pinned state tokenizer: https://github.com/robocasa-benchmark/openpi/blob/5a6beda9ff99da30b4e1b59320f6a32971d7c397/src/openpi/models/tokenizer.py
- normalization guidance: https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md
