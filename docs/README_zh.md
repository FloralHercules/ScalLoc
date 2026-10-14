# ScalLoc 中文使用说明

本项目整理自原来可以运行的 **ScalLoc 快版本**，数据集公开名称为 **Thermal-Oblique**。不是 `ScalLoc-v1_end`，也不是之后的端到端位姿校正原型。

这里只发布代码与配置，不包含数据集、卫星图、DSM、权重或历史结果。重构的目标是保持计算逻辑不变；实际对照测试状态请看 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，不能把“代码已整理”当作“数值已验证一致”。

## 安装与运行

在独立 Python 环境中先安装匹配的 PyTorch/torchvision，再安装依赖：

```bash
python -m pip install -r requirements.txt
```

原本地参考环境为 Python 3.12.9、PyTorch 2.6.0+cu126、torchvision 0.21.0+cu126、RTX 4060 Ti 16 GB。不同硬件、驱动或依赖版本可能造成数值差异；这不是最低硬件要求声明。

按 [DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) 放置外部资源，然后运行：

默认快版**不读取 DSM，不需要下载 `dsm.tif` 就能运行**。配置中保留 `paths.dsm_map` 作为兼容预留路径即可，该路径对应的文件可以不存在；卫星参考图 `ref.tif` 仍然必需。

```bash
# 只检查资源，不加载模型。
python run.py --config configs/default.yaml --check

# 默认跑完整曲线序列，不是全部 216 条。
python run.py --config configs/default.yaml

# 只跑一个窗口检查流程，不代表完整轨迹成绩。
python run.py --config configs/default.yaml --max-windows 1

# 指定序列；需要多条时可以重复 --dataset。
python run.py --config configs/default.yaml --dataset Rural_Day_Curve_H150_P30_S01

# 明确要求才运行全部可用序列。
python run.py --config configs/default.yaml --all-datasets
```

默认序列 `Rural_Day_Curve_H150_P30_S01` 在参考数据中为 138 帧，`max_windows: 0` 表示完整运行。指定 `--output-dir outputs/my_run` 可以改变输出目录；已有非空目录会被拒绝覆盖。

## 磁盘空间不足时

不必再次复制数据或权重。在 `configs/local.example.yaml` 中把占位路径换成已有资源的实际路径：

```bash
python run.py --config configs/local.example.yaml --check
python run.py --config configs/local.example.yaml
```

所有相对路径以**项目根目录**为基准，而不是配置文件所在目录。个人本地配置不要上传，公开仓库只保留占位示例。Hugging Face 与百度网盘下载地址尚待作者补充，不提供虚构链接。

## 保持原实验条件

默认保留原快版的 12 帧窗口、1 帧重叠、固定内参轨迹 BA、锚点逻辑，以及原实现使用的后续帧俯仰角/相对高度先验。它**不是纯 VGGT、不是仅首帧先验实验**。

扰动默认关闭：`run.pitch_max_deg: 0`、`run.height_max_fraction: 0`；`run.vggt_predictions: false`。详细中间图默认不保存。比较重构前后结果时，权重、输入数据、地图、帧顺序、参数和运行环境都应保持一致。

## 发布前注意

本次重构已验证一条完整曲线：138 帧、13 个窗口，与重新运行的原版及
10 月 2 日保存的原版结果均逐项一致，逐帧 CSV 的 SHA256 也一致。
详细指标、比较范围和复现命令见 [复现报告](REPRODUCIBILITY.md)。
这不代表已经复核全部 216 条轨迹。

- 数据、权重、结果、缓存、个人绝对路径不属于代码发布包。
- 自有代码根许可证尚待作者选择，见 [LICENSE_NOTICE.md](../LICENSE_NOTICE.md)。
- 第三方许可证已单独保留，见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)；不能把整个项目及所有权重笼统标成 MIT。
- 更完整的结构和配置说明见 [英文 README](../README.md)。
