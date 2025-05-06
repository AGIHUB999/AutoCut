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
- **Low-memory mode** for processing large video files
- **FFmpeg-based implementation** for improved compatibility and performance
- **Vertical video conversion** for short-video platforms (TikTok, Instagram, etc.)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage

```bash
python autocut.py input_video.mp4 output_video.mp4
```

#### For Large Video Files (FFmpeg Version)

```bash
python autocut_ffmpeg.py input_video.mp4 output_video.mp4
```

#### Convert to Vertical Format

```bash
python vertical_converter.py input_video.mp4 vertical_output.mp4
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
- `--low-memory`: Enable low-memory mode for large video files (default: auto)
- `--chunk-size`: Size of chunks for processing in low-memory mode (seconds), default 300

#### Vertical Converter Parameters

```bash
python vertical_converter.py input_video.mp4 vertical_output.mp4 --focus face --blur 50 --caption "Concert Highlights"
```

Parameter description:
- `--width`: Output video width, default 1080
- `--height`: Output video height, default 1920
- `--focus`: Focus mode (auto/center/face/motion), default auto
- `--blur`: Background blur level (0-100), default 30
- `--bg-color`: Background color, default black
- `--zoom`: Zoom factor, default 1.2
- `--quality`: Output quality (low/medium/high), default medium
- `--caption`: Add caption text to the video
- `--gpu`: Use GPU acceleration if available

### Examples

```bash
# Extract 10 highlight clips, each clip minimum 3 seconds, maximum 20 seconds
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# Lower the threshold to detect more potential highlight moments
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5

# Process a large video file with FFmpeg version
python autocut_ffmpeg.py large_concert.mp4 highlights.mp4 --volume-threshold 0.8 --scene-threshold 0.3

# Complete workflow: extract highlights and convert to vertical format
python autocut_ffmpeg.py concert.mp4 highlights.mp4
python vertical_converter.py highlights.mp4 vertical_highlights.mp4 --focus face --caption "Amazing Concert"
```

### How It Works

1. **Video Import and Processing**: Load video and extract audio
2. **Audio Analysis**: Analyze audio features including energy, spectral contrast, rhythm changes, etc.
3. **Highlight Detection**: Detect highlight moments based on comprehensive scoring of multiple features
4. **Intelligent Editing**: Select the highest-scoring clips, add fade-in/fade-out effects, generate highlight compilation
5. **Visualization Analysis**: Generate analysis charts showing the detection process and results
6. **Vertical Conversion** (optional): Convert horizontal video to vertical format for short-video platforms

### Versions

#### Standard Version (`autocut.py`)
- Uses Python libraries for audio and video processing
- Best for regular-sized videos (up to ~1 hour)
- Includes low-memory mode for larger files

#### FFmpeg Version (`autocut_ffmpeg.py`)
- Uses FFmpeg for audio and video processing
- Optimized for very large video files (multiple hours)
- Significantly reduced memory usage
- More robust error handling and recovery
- Requires FFmpeg to be installed on your system

#### Vertical Converter (`vertical_converter.py`)
- Converts horizontal videos to vertical format (9:16 aspect ratio)
- Intelligent focus detection (faces, motion)
- Background blur and custom styling options
- Optimized for short-video platforms

### Notes

- Processing large video files may require significant time
- High-quality video sources are recommended for better detection results
- Parameters may need adjustment for different types of concerts to achieve optimal results
- For videos larger than 2 hours, use the FFmpeg version
- Vertical conversion works best with videos that have clear focal points

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
- **低内存模式**，可处理大型视频文件
- **基于FFmpeg的实现**，提高兼容性和性能
- **竖屏视频转换**，适配抖音、快手等短视频平台

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

#### 基本用法

```bash
python autocut.py 输入视频.mp4 输出视频.mp4
```

#### 处理大型视频文件（FFmpeg版本）

```bash
python autocut_ffmpeg.py 输入视频.mp4 输出视频.mp4
```

#### 转换为竖屏格式

```bash
python vertical_converter.py 输入视频.mp4 竖屏输出.mp4
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
- `--low-memory`: 启用低内存模式处理大型视频文件（默认：自动）
- `--chunk-size`: 低内存模式下的处理块大小（秒），默认300

#### 竖屏转换参数

```bash
python vertical_converter.py 输入视频.mp4 竖屏输出.mp4 --focus face --blur 50 --caption "演唱会精彩片段"
```

参数说明：
- `--width`: 输出视频宽度，默认1080
- `--height`: 输出视频高度，默认1920
- `--focus`: 焦点模式（auto/center/face/motion），默认auto
- `--blur`: 背景模糊程度（0-100），默认30
- `--bg-color`: 背景颜色，默认black
- `--zoom`: 放大因子，默认1.2
- `--quality`: 输出质量（low/medium/high），默认medium
- `--caption`: 添加标题文本
- `--gpu`: 使用GPU加速（如果可用）

### 示例

```bash
# 提取10个高潮片段，每个片段最短3秒，最长20秒
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# 降低阈值，检测更多可能的高潮部分
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5

# 使用FFmpeg版本处理大型视频文件
python autocut_ffmpeg.py large_concert.mp4 highlights.mp4 --volume-threshold 0.8 --scene-threshold 0.3

# 完整工作流：提取高潮片段并转换为竖屏格式
python autocut_ffmpeg.py concert.mp4 highlights.mp4
python vertical_converter.py highlights.mp4 vertical_highlights.mp4 --focus face --caption "精彩演唱会"
```

### 工作原理

1. **视频导入和处理**：加载视频并提取音频
2. **音频分析**：分析音频特征，包括能量、频谱对比度、节奏变化等
3. **高潮检测**：基于多种特征综合评分，检测视频中的高潮部分
4. **智能剪辑**：选择评分最高的片段，添加淡入淡出效果，生成精彩集锦
5. **可视化分析**：生成分析图表，展示检测过程和结果
6. **竖屏转换**（可选）：将横屏视频转换为适合短视频平台的竖屏格式

### 版本说明

#### 标准版本 (`autocut.py`)
- 使用Python库进行音频和视频处理
- 适合常规大小的视频（最长约1小时）
- 包含低内存模式，可处理较大文件

#### FFmpeg版本 (`autocut_ffmpeg.py`)
- 使用FFmpeg进行音频和视频处理
- 专为超大型视频文件优化（数小时长度）
- 显著降低内存使用量
- 更强大的错误处理和恢复机制
- 需要系统安装FFmpeg

#### 竖屏转换器 (`vertical_converter.py`)
- 将横屏视频转换为竖屏格式（9:16比例）
- 智能焦点检测（人脸、运动区域）
- 背景模糊和自定义样式选项
- 为短视频平台优化

### 注意事项

- 处理大型视频文件可能需要较长时间
- 推荐使用高质量的视频源以获得更好的检测效果
- 对于不同类型的演唱会，可能需要调整参数以获得最佳效果
- 对于超过2小时的视频，建议使用FFmpeg版本
- 竖屏转换对有明确焦点的视频效果最佳

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
- **Mode faible mémoire** pour traiter les fichiers vidéo volumineux
- **Implémentation basée sur FFmpeg** pour une compatibilité et des performances améliorées
- **Conversion en format vertical** pour les plateformes de vidéos courtes (TikTok, Instagram, etc.)

### Installation

```bash
pip install -r requirements.txt
```

### Utilisation

#### Utilisation de base

```bash
python autocut.py video_entree.mp4 video_sortie.mp4
```

#### Pour les fichiers vidéo volumineux (version FFmpeg)

```bash
python autocut_ffmpeg.py video_entree.mp4 video_sortie.mp4
```

#### Conversion en format vertical

```bash
python vertical_converter.py video_entree.mp4 video_verticale.mp4
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
- `--low-memory` : Activer le mode faible mémoire pour les fichiers vidéo volumineux (par défaut : auto)
- `--chunk-size` : Taille des morceaux pour le traitement en mode faible mémoire (secondes), par défaut 300

#### Paramètres du convertisseur vertical

```bash
python vertical_converter.py video_entree.mp4 video_verticale.mp4 --focus face --blur 50 --caption "Moments forts du concert"
```

Description des paramètres :
- `--width` : Largeur de la vidéo de sortie, par défaut 1080
- `--height` : Hauteur de la vidéo de sortie, par défaut 1920
- `--focus` : Mode de focus (auto/center/face/motion), par défaut auto
- `--blur` : Niveau de flou d'arrière-plan (0-100), par défaut 30
- `--bg-color` : Couleur d'arrière-plan, par défaut black
- `--zoom` : Facteur de zoom, par défaut 1,2
- `--quality` : Qualité de sortie (low/medium/high), par défaut medium
- `--caption` : Ajouter un texte de légende à la vidéo
- `--gpu` : Utiliser l'accélération GPU si disponible

### Exemples

```bash
# Extraire 10 clips forts, chaque clip minimum 3 secondes, maximum 20 secondes
python autocut.py concert.mp4 highlights.mp4 --min-duration 3 --max-duration 20 --count 10

# Abaisser le seuil pour détecter plus de moments forts potentiels
python autocut.py concert.mp4 highlights.mp4 --threshold 0.5

# Traiter un fichier vidéo volumineux avec la version FFmpeg
python autocut_ffmpeg.py large_concert.mp4 highlights.mp4 --volume-threshold 0.8 --scene-threshold 0.3

# Flux de travail complet : extraire les moments forts et convertir en format vertical
python autocut_ffmpeg.py concert.mp4 highlights.mp4
python vertical_converter.py highlights.mp4 vertical_highlights.mp4 --focus face --caption "Concert Incroyable"
```

### Comment ça fonctionne

1. **Importation et traitement vidéo** : Chargement de la vidéo et extraction de l'audio
2. **Analyse audio** : Analyse des caractéristiques audio, y compris l'énergie, le contraste spectral, les changements de rythme, etc.
3. **Détection des moments forts** : Détection des moments forts basée sur une notation complète de plusieurs caractéristiques
4. **Édition intelligente** : Sélection des clips les mieux notés, ajout d'effets de fondu, génération d'une compilation de moments forts
5. **Analyse visuelle** : Génération de graphiques d'analyse montrant le processus de détection et les résultats
6. **Conversion verticale** (optionnelle) : Conversion de la vidéo horizontale en format vertical pour les plateformes de vidéos courtes

### Versions

#### Version standard (`autocut.py`)
- Utilise des bibliothèques Python pour le traitement audio et vidéo
- Idéal pour les vidéos de taille normale (jusqu'à environ 1 heure)
- Inclut un mode faible mémoire pour les fichiers plus volumineux

#### Version FFmpeg (`autocut_ffmpeg.py`)
- Utilise FFmpeg pour le traitement audio et vidéo
- Optimisé pour les fichiers vidéo très volumineux (plusieurs heures)
- Utilisation de la mémoire considérablement réduite
- Gestion des erreurs et récupération plus robustes
- Nécessite que FFmpeg soit installé sur votre système

#### Convertisseur vertical (`vertical_converter.py`)
- Convertit les vidéos horizontales en format vertical (rapport d'aspect 9:16)
- Détection intelligente de la zone d'intérêt (visages, mouvement)
- Options de flou d'arrière-plan et de style personnalisé
- Optimisé pour les plateformes de vidéos courtes

### Remarques

- Le traitement de fichiers vidéo volumineux peut nécessiter un temps considérable
- Des sources vidéo de haute qualité sont recommandées pour de meilleurs résultats de détection
- Les paramètres peuvent nécessiter des ajustements pour différents types de concerts afin d'obtenir des résultats optimaux
- Pour les vidéos de plus de 2 heures, utilisez la version FFmpeg
- La conversion verticale fonctionne mieux avec des vidéos qui ont des points focaux clairs

---

## 内容生成器 (Content Generator)

内容生成器是一个强大的工具，可以自动分析视频内容，并生成吸引眼球的标题和封面，帮助您的视频在短视频平台上获得更多曝光。

### 功能

- **语音识别**：从视频中提取语音内容，转换为文本
- **OCR文本识别**：识别视频中出现的文字
- **智能标题生成**：根据视频内容自动生成多个吸引眼球的标题
- **精美封面制作**：选择最佳视频帧，添加文字和特效，生成多个候选封面
- **多种风格选择**：支持多种标题和封面风格，满足不同平台和受众需求

### 使用方法

```bash
python content_generator.py input_video.mp4 [options]
```

#### 基本参数

- `input`: 输入视频文件路径（必需）
- `-o, --output`: 输出目录路径（可选，默认为视频文件所在目录）

#### 高级参数

- `--frame-interval`: 截取帧的间隔（秒），默认为5
- `--no-speech`: 禁用语音识别
- `--no-ocr`: 禁用OCR文本识别
- `--title-style`: 标题风格，可选值: exciting, funny, emotional, informative
- `--cover-style`: 封面风格，可选值: modern, minimal, dramatic, colorful
- `--language`: 语言，可选值: auto, zh, en
- `--max-title-length`: 标题最大长度，默认为50
- `--api-key`: OpenAI API密钥（用于更高质量的标题生成）
- `--font-path`: 自定义字体路径
- `--use-gpu`: 使用GPU加速（如果可用）
- `--thumbnail-count`: 生成的候选封面数量，默认为5
- `--title-count`: 生成的候选标题数量，默认为3

### 示例

生成中文风格的标题和封面：

```bash
python content_generator.py concert.mp4 --language zh --title-style exciting --cover-style dramatic
```

使用OpenAI API生成更高质量的标题：

```bash
python content_generator.py interview.mp4 --api-key YOUR_OPENAI_API_KEY --title-count 5
```

### 依赖

- OpenCV: `pip install opencv-python`
- Pillow: `pip install pillow`
- SpeechRecognition: `pip install SpeechRecognition`
- Pytesseract: `pip install pytesseract`
- FFmpeg: 必须安装并添加到系统PATH

### 输出

内容生成器会在输出目录中创建以下文件：

- `titles.txt`: 生成的标题列表
- `thumbnails/`: 包含生成的封面图像的目录
- `transcription.txt`: 视频的语音转录文本（如果启用）
- `ocr_text.txt`: 视频中识别的文本（如果启用）
- `metadata.json`: 包含视频信息和生成内容的元数据

## 完整工作流示例

以下是使用AutoCut工具链处理演唱会视频的完整工作流示例：

1. **提取高潮片段**：
```bash
python autocut_ffmpeg.py concert_full.mp4 --output concert_highlights.mp4 --threshold 0.8 --min-clip-length 15 --max-clip-length 30
```

2. **转换为竖屏格式**：
```bash
python vertical_converter.py concert_highlights.mp4 --output concert_vertical.mp4 --focus-mode face --blur 10 --zoom 1.2
```

3. **生成标题和封面**：
```bash
python content_generator.py concert_vertical.mp4 --title-style exciting --cover-style dramatic --language zh
```

完成这三个步骤后，您将获得：
- 一个包含演唱会精彩片段的横屏视频
- 一个针对短视频平台优化的竖屏版本
- 多个吸引眼球的标题选项
- 多个精美的封面图像选项

这样，您就可以选择最佳的标题和封面，将视频上传到短视频平台，吸引更多观众！