# Review-13 六臂脚本与归档结果的对应关系

权威交付物：`archives/REVIEW13_DELIVERY_20261103.tar.gz`。六臂 M1–M6，M0 使用同一服务器项目中 Review-12 seed43 BSAM 的权重。Review-12、Review-13 及后续 Review-12 resume 属于同一服务器项目，分开归档只是交付形式。所附脚本依据早期规划整理，不是服务器运行脚本的逐字节镜像；Review-13 交付归档没有附上当时的完整训练入口，不能据此推断实验或原服务器上的脚本不存在。

| 臂 | 当前定义 | 归档 Test RMSE (Mg/ha) |
| --- | --- | ---: |
| M1 | S2 与采集日期置零 | 62.8947 |
| M2 | DEM patch 与地形 metadata 置零 | 60.3787 |
| M3 | 土地覆盖置零 | 58.1394 |
| M4 | ALOS 归一化值与原始 DN 置零 | 58.1887 |
| M5 | 跳过 FiLM 条件机制 | 58.9326 |
| M6 | 全部八维 metadata 置零，FiLM 机制保留 | 58.6989 |

`ABLATION_CONFIGS/PROTOCOL_TEMPLATE.json` 仍是禁用执行的规划模板。静态检查仅核对代码形状和配置；训练、权重加载与数据处理链均须另外验证。不要把所附脚本的哈希当作归档的服务器源码哈希。
