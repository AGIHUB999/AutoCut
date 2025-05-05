# AutoCut - 演唱会高潮自动剪辑工具

AutoCut是一个基于AI的视频自动剪辑工具，专为自媒体博主设计，可以自动检测并剪辑演唱会视频中的精彩高潮部分。

## 功能特点

- 自动检测视频中的高潮部分（基于音量、节奏、观众反应等）
- 智能选择最精彩的片段并生成集锦视频
- 支持掌声检测，更准确地识别观众反应热烈的部分
- 节奏变化检测，捕捉音乐高潮
- 可视化分析结果，直观展示检测过程
- 支持自定义参数，灵活调整剪辑效果

## 安装依赖

```bash
pip install opencv-python librosa matplotlib moviepy pydub torch numpy scipy
```

## 使用方法

### 基本用法

```bash
python autocut.py 输入视频.mp4 输出视频.mp4
```

### 高级参数

```bash
python autocut.py 输入视频.mp4 输出视频.mp4 --min-duration 5 --max-duration 30 --threshold 0.7 --count 5
```

参数说明：
- `--min-duration`: 最小片段时长(秒)，默认5秒
- `--max-duration`: 最大片段时长(秒)，默认30秒
- `--threshold`: 能量阈值(0-1)，默认0.7
- `--count`: 要提取的高潮片段数量，默认5个
- `--no-applause`: 禁用掌声检测
- `--no-tempo`: 禁用节奏变化检测

## 示例

```bash
# 提取10个高潮片段，每个片段最短3秒，最长20秒
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# 降低阈值，检测更多可能的高潮部分
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5
```

## 工作原理

1. **视频导入和处理**：加载视频并提取音频
2. **音频分析**：分析音频特征，包括能量、频谱对比度、节奏变化等
3. **高潮检测**：基于多种特征综合评分，检测视频中的高潮部分
4. **智能剪辑**：选择评分最高的片段，添加淡入淡出效果，生成精彩集锦
5. **可视化分析**：生成分析图表，展示检测过程和结果

## 注意事项

- 处理大型视频文件可能需要较长时间
- 推荐使用高质量的视频源以获得更好的检测效果
- 对于不同类型的演唱会，可能需要调整参数以获得最佳效果