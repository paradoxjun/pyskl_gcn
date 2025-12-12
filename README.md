# pyskl_gcn

**Keypoint / Skeleton Action Recognition — A simplified, efficient training framework.**  
作者：**paradoxjun**（抖音 / B站 / 小红书 / CSDN 同名：`paradoxjun`）

**代码解释**：https://blog.csdn.net/qq_40387714/article/details/155860538

**百度网盘**：https://pan.baidu.com/share/init?surl=2HvEYPVwA_C4JICYvFDqBg&pwd=c6tn

---

## 中文介绍

本项目面向**基于关键点（骨架/3D关键点）的人体行为识别**，提供一套**简化、高效、易复现**的训练框架。项目在 **NTU-RGB+D60** 数据集上对 **ST-GCN++** 等模型实现进行了针对性优化，在仅使用**3D关键点单流输入**的情况下取得 **91.64% Top-1 Accuracy**，优于原论文与部分开源实现。

本仓库强调“**简化输入特征与推理流程**”，并在工程实践角度指出：当前研究中**过度追求多流融合**与**复杂采样策略**往往带来更高的实现成本与部署复杂度，但对真实业务落地（低延迟、稳定、可维护）并不总是最优选择。本项目也为后续结合 **YOLO-Pose** 构建实时动作识别管线打下基础。

---

## 亮点

- **单流 3D 关键点**：优先把输入与推理做“够用且简单”
- **高复现性**：训练/预处理/评估链路清晰，便于定位效果差异来源
- **轻量依赖**：核心仅依赖 `torch`、`numpy` 等基础库，降低环境配置难度
- **工程导向**：为后续实时推理（结合 YOLO-Pose）保留清晰的扩展路径

---

## 结果

- 数据集：**NTU-RGB+D60**
- 输入：**3D skeleton / keypoints（单流）**
- 模型：**ST-GCN++（及相关变体）**
- 指标：**Top-1 Accuracy**
- 最佳结果：**91.64%**

> 注：不同数据划分、预处理细节、训练策略都会显著影响最终精度。本项目的目标是在“更低复杂度”前提下拿到强基线。

---

## 数据格式（NPZ）

项目使用 NPZ 管线时，常见字段如下（按样本存储为 object 数组，每个样本是一个可变长序列）：

- `data[idx]`：`(F, P, V, 3)`  
  - `F`：帧数  
  - `P`：人数（常用 1 或 2）  
  - `V`：关节点数（如 NTU 25 joints）  
  - `3`：`(x, y, z)`  
- `label[idx]`：类别 ID
- `view[idx]`（可选）：视角 ID
- `valid_frame[idx]`（可选）：有效帧数
- `xyz_sphere[idx]`（可选）：`(F, P, 4)`，每帧每人 `(cx, cy, cz, r)`（用于可视化/归一化等）

---

## 依赖与安装

建议 Python 3.8+。

最小核心依赖：
- `torch`
- `numpy`

安装示例：
```bash
pip install torch numpy
