# AutoCut - AI Concert Highlight Editing Tool

[English](#english) | [中文](#chinese) | [Français](#french)

<a id="english"></a>
## English

AutoCut is an AI-based video editing tool designed for content creators, capable of automatically detecting and editing highlight moments from concert videos.

### Features

- Automatic detection of highlight moments (based on volume, rhythm, audience reaction, etc.)
- Intelligent selection of the most exciting clips and generation of highlight videos
- Applause detection for more accurate identification of moments with enthusiastic audience reactions
- Rhythm change detection to capture musical climaxes
- Visualization of analysis results for intuitive display of the detection process
- Customizable parameters for flexible editing effects

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage

```bash
python autocut.py input_video.mp4 output_video.mp4
```

#### Advanced Parameters

```bash
python autocut.py input_video.mp4 output_video.mp4 --min-duration 5 --max-duration 30 --threshold 0.7 --count 5
```

Parameter description:
- `--min-duration`: Minimum clip duration (seconds), default 5 seconds
- `--max-duration`: Maximum clip duration (seconds), default 30 seconds
- `--threshold`: Energy threshold (0-1), default 0.7
- `--count`: Number of highlight clips to extract, default 5
- `--no-applause`: Disable applause detection
- `--no-tempo`: Disable rhythm change detection

### Examples

```bash
# Extract 10 highlight clips, each clip minimum 3 seconds, maximum 20 seconds
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# Lower the threshold to detect more potential highlight moments
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5
```

### How It Works

1. **Video Import and Processing**: Load video and extract audio
2. **Audio Analysis**: Analyze audio features including energy, spectral contrast, rhythm changes, etc.
3. **Highlight Detection**: Detect highlight moments based on comprehensive scoring of multiple features
4. **Intelligent Editing**: Select the highest-scoring clips, add fade-in/fade-out effects, generate highlight compilation
5. **Visualization Analysis**: Generate analysis charts showing the detection process and results

### Notes

- Processing large video files may require significant time
- High-quality video sources are recommended for better detection results
- Parameters may need adjustment for different types of concerts to achieve optimal results

---

<a id="chinese"></a>
## 中文

AutoCut是一个基于AI的视频自动剪辑工具，专为内容创作者设计，可以自动检测并剪辑演唱会视频中的精彩高潮部分。

### 功能特点

- 自动检测视频中的高潮部分（基于音量、节奏、观众反应等）
- 智能选择最精彩的片段并生成集锦视频
- 支持掌声检测，更准确地识别观众反应热烈的部分
- 节奏变化检测，捕捉音乐高潮
- 可视化分析结果，直观展示检测过程
- 支持自定义参数，灵活调整剪辑效果

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

#### 基本用法

```bash
python autocut.py 输入视频.mp4 输出视频.mp4
```

#### 高级参数

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

### 示例

```bash
# 提取10个高潮片段，每个片段最短3秒，最长20秒
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# 降低阈值，检测更多可能的高潮部分
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5
```

### 工作原理

1. **视频导入和处理**：加载视频并提取音频
2. **音频分析**：分析音频特征，包括能量、频谱对比度、节奏变化等
3. **高潮检测**：基于多种特征综合评分，检测视频中的高潮部分
4. **智能剪辑**：选择评分最高的片段，添加淡入淡出效果，生成精彩集锦
5. **可视化分析**：生成分析图表，展示检测过程和结果

### 注意事项

- 处理大型视频文件可能需要较长时间
- 推荐使用高质量的视频源以获得更好的检测效果
- 对于不同类型的演唱会，可能需要调整参数以获得最佳效果

---

<a id="french"></a>
## Français

AutoCut est un outil d'édition vidéo basé sur l'IA, conçu pour les créateurs de contenu, capable de détecter et d'éditer automatiquement les moments forts des vidéos de concert.

### Fonctionnalités

- Détection automatique des moments forts (basée sur le volume, le rythme, la réaction du public, etc.)
- Sélection intelligente des clips les plus excitants et génération de vidéos de moments forts
- Détection des applaudissements pour une identification plus précise des moments avec des réactions enthousiastes du public
- Détection des changements de rythme pour capturer les apogées musicales
- Visualisation des résultats d'analyse pour un affichage intuitif du processus de détection
- Paramètres personnalisables pour des effets d'édition flexibles

### Installation

```bash
pip install -r requirements.txt
```

### Utilisation

#### Utilisation de base

```bash
python autocut.py video_entree.mp4 video_sortie.mp4
```

#### Paramètres avancés

```bash
python autocut.py video_entree.mp4 video_sortie.mp4 --min-duration 5 --max-duration 30 --threshold 0.7 --count 5
```

Description des paramètres :
- `--min-duration` : Durée minimale du clip (secondes), par défaut 5 secondes
- `--max-duration` : Durée maximale du clip (secondes), par défaut 30 secondes
- `--threshold` : Seuil d'énergie (0-1), par défaut 0,7
- `--count` : Nombre de clips forts à extraire, par défaut 5
- `--no-applause` : Désactiver la détection des applaudissements
- `--no-tempo` : Désactiver la détection des changements de rythme

### Exemples

```bash
# Extraire 10 clips forts, chaque clip minimum 3 secondes, maximum 20 secondes
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# Abaisser le seuil pour détecter plus de moments forts potentiels
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5
```

### Comment ça fonctionne

1. **Importation et traitement vidéo** : Chargement de la vidéo et extraction de l'audio
2. **Analyse audio** : Analyse des caractéristiques audio, y compris l'énergie, le contraste spectral, les changements de rythme, etc.
3. **Détection des moments forts** : Détection des moments forts basée sur une notation complète de plusieurs caractéristiques
4. **Édition intelligente** : Sélection des clips les mieux notés, ajout d'effets de fondu, génération d'une compilation de moments forts
5. **Analyse visuelle** : Génération de graphiques d'analyse montrant le processus de détection et les résultats

### Remarques

- Le traitement de fichiers vidéo volumineux peut nécessiter un temps considérable
- Des sources vidéo de haute qualité sont recommandées pour de meilleurs résultats de détection
- Les paramètres peuvent nécessiter des ajustements pour différents types de concerts afin d'obtenir des résultats optimaux