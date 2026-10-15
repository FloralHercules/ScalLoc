# ScalLoc 中文使用说明

**ScalLoc** 面向连续斜视热成像无人机定位，通过窗口重建、轨迹优化、卫星地图锚定和跨窗口传播，恢复全局轨迹。只需要二维卫星参考图，无需 DSM 或预建三维地图。

[论文（arXiv:2610.14866）](https://arxiv.org/abs/2610.14866) · [数据与权重（Hugging Face）](https://huggingface.co/datasets/FloralHercules/Thermal-Oblique) · [英文主页](../README.md)

![ScalLoc 方法框架](../assets/ScalLoc.png)

## Thermal-Oblique 数据集

数据集包含 **216 条轨迹、28,112 帧热成像图像**及逐帧标注，覆盖城市与乡村、昼间与夜间，以及直线、曲线和环绕三种运动类型。

![Thermal-Oblique 数据集概览](../assets/Thermal-Oblique.png)

数据集、卫星参考图和三个模型权重已发布于 [Hugging Face](https://huggingface.co/datasets/FloralHercules/Thermal-Oblique)，可直接下载，不放入 GitHub 代码仓库。百度网盘镜像待补充。

资源包下载后，将其中的 `data/` 和 `weights/` 合并到项目根目录；具体路径与校验值见 [资源指南](DATA_AND_WEIGHTS.md)。磁盘空间有限时，也可解压到其他盘，通过配置指定路径。

## 安装与运行

在独立 Python 环境中先安装匹配的 PyTorch/torchvision，再安装依赖：

```bash
python -m pip install -r requirements.txt
```

按 [DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) 放置外部资源。

默认流程**不读取 DSM，不需要下载 `dsm.tif` 就能运行**。配置中保留 `paths.dsm_map` 作为兼容预留路径即可，该路径对应的文件可以不存在；卫星参考图 `ref.tif` 仍然必需。

如需在其他流程中使用 DSM，可从 [Thermal-UAV 资源页](https://huggingface.co/datasets/FloralHercules/Thermal-UAV/tree/main) 获取，并放置为 `data/maps/changsha/dsm.tif`。来源文档标注该 DSM 为 5.29 m/像素，与本项目的 0.3 m/像素卫星地图并非同一网格，也未验证为可直接替换的资源；使用前需核对实际 GeoTIFF 的覆盖范围、坐标系和分辨率，完成必要的配准并匹配配置。详见[资源指南](DATA_AND_WEIGHTS.md)。

```bash
# 只检查资源，不加载模型。
python run.py --config configs/default.yaml --check

# 默认跑完整曲线序列，不是全部 216 条。
python run.py --config configs/default.yaml

# 只跑一个窗口检查流程，不代表完整轨迹成绩。
python run.py --config configs/default.yaml --max-windows 1

# 指定序列；需要多条时可以重复 --dataset。
python run.py --config configs/default.yaml --dataset Rural_Day_Curve_H150_P30_S01

# 运行全部可用序列。
python run.py --config configs/default.yaml --all-datasets
```

默认序列 `Rural_Day_Curve_H150_P30_S01` 在参考数据中为 138 帧，`max_windows: 0` 表示完整运行。指定 `--output-dir outputs/my_run` 可以改变输出目录；已有非空目录会被拒绝覆盖。

## 磁盘空间不足时

不必再次复制数据或权重。在 `configs/local.example.yaml` 中把占位路径换成已有资源的实际路径：

```bash
python run.py --config configs/local.example.yaml --check
python run.py --config configs/local.example.yaml
```

所有相对路径以**项目根目录**为基准，而不是配置文件所在目录。个人本地配置不要上传，公开仓库只保留占位示例。

## 默认配置

默认配置采用 12 帧窗口、1 帧重叠、固定内参轨迹优化、地图锚定和跨窗口传播，并使用元数据中的逐帧俯仰角与相对高度先验。

扰动默认关闭：`run.pitch_max_deg: 0`、`run.height_max_fraction: 0`；`run.vggt_predictions: false`。详细中间图默认不保存。

## 前置工作与引用

[SCC-Loc](https://arxiv.org/abs/2604.03120) 是我们在无人机热成像地理定位方向的前置工作，提出统一的语义级联共识框架。

如果在研究中使用 ScalLoc 或 Thermal-Oblique，请引用：

```bibtex
@article{zhang2026scalloc,
  title   = {ScalLoc: Continuous Oblique Thermal UAV Geo-Localization with a 2D Satellite Map},
  author  = {Zhang, Xiaoran and Liu, Yu and Liang, Jinyu and Xiao, Huaxin},
  journal = {arXiv preprint arXiv:2610.14866},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.14866}
}
```

相关的 SCC-Loc 工作请引用：

```bibtex
@article{zhang2026sccloc,
  author={Zhang, Xiaoran and Liu, Yu and Liang, Jinyu and Li, Kangqiushi and Huang, Zhiwei and Xiao, Huaxin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  title={SCC-Loc: A Unified Semantic Cascade Consensus Framework for UAV Thermal Geo-Localization},
  year={2026},
  volume={19},
  number={},
  pages={25299--25315}
}
```

## 许可证与资源说明

- 数据、权重、结果、缓存、个人绝对路径不属于代码发布包。
- 自有代码根许可证尚待作者选择，见 [LICENSE_NOTICE.md](../LICENSE_NOTICE.md)。
- 第三方许可证已单独保留，见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)；不能把整个项目及所有权重笼统标成 MIT。
- 更完整的结构和配置说明见 [英文 README](../README.md)。
