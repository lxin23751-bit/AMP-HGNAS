# 节点分类可视化功能说明

## 功能概述

代码已添加节点嵌入可视化功能，可以生成类似论文中的t-SNE/UMAP降维可视化图。

## 生成的可视化文件

运行训练脚本后，会在 `./visualizations/{dataset}/` 目录下自动生成以下文件：

### 1. 节点嵌入可视化（主要功能）
- `embedding_tsne_{dataset}_seed{seed}_test.png` - 测试集的t-SNE可视化
- `embedding_tsne_{dataset}_seed{seed}_val.png` - 验证集的t-SNE可视化
- `embedding_umap_{dataset}_seed{seed}_test.png` - 测试集的UMAP可视化（如果安装了umap-learn）
- `embedding_umap_{dataset}_seed{seed}_val.png` - 验证集的UMAP可视化

### 2. 其他可视化
- `confusion_matrix_*.png` - 混淆矩阵
- `class_performance_*.png` - 每个类别的性能柱状图
- `accuracy_curves_*.png` - 训练过程准确率曲线
- `classification_report_*.txt` - 详细的分类报告

## 使用方法

直接运行训练脚本即可，可视化会自动生成：

```bash
python main_path.py --dataset DBLP --num-hops 6 --residual --amp --seeds 1 --arch dblp
```

## 依赖安装

如果需要使用UMAP可视化（可选，t-SNE是默认的）：

```bash
pip install umap-learn
```

其他依赖都已包含在代码中：
- matplotlib
- seaborn
- scikit-learn

## 可视化效果

生成的t-SNE/UMAP图会显示：
- 不同类别的节点用不同颜色表示
- 每个点代表一个节点
- 如果同类节点聚集在一起，说明模型学习到了好的表示
- 不同类别的节点分离越清晰，说明模型性能越好
