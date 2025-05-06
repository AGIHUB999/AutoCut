#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Subtitle Generator - 为视频自动生成字幕，支持中文
"""

import os
import sys
import json
import tempfile
import logging
import time
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Optional
import subprocess
import whisper
from tqdm import tqdm
import re
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SubtitleGenerator")

class SubtitleGenerator:
    """为视频自动生成字幕的工具类"""
    
    def __init__(self, input_path: str, output_path: str = None, config: Dict = None):
        """
        初始化SubtitleGenerator
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径，默认为原文件名+_subtitled后缀
            config: 配置参数
        """
        self.input_path = input_path
        
        # 如果未指定输出路径，使用默认命名
        if output_path is None:
            input_dir = os.path.dirname(os.path.abspath(input_path))
            input_filename = os.path.basename(input_path)
            name, ext = os.path.splitext(input_filename)
            self.output_path = os.path.join(input_dir, f"{name}_subtitled{ext}")
        else:
            self.output_path = output_path
        
        # 默认配置
        self.config = {
            "model_size": "medium",      # whisper模型大小: tiny, base, small, medium, large
            "language": "auto",          # 语言: auto, zh, en 等
            "font_size": 24,             # 字幕字体大小
            "font_color": "white",       # 字幕颜色
            "stroke_color": "black",     # 字幕描边颜色
            "stroke_width": 1.5,         # 字幕描边宽度
            "position": "bottom",        # 字幕位置: bottom, top
            "align": "center",           # 字幕对齐方式: center, left, right
            "max_line_width": 40,        # 每行最大字符数
            "temp_dir": tempfile.mkdtemp(),  # 临时目录
            "srt_output": True,          # 是否同时输出SRT文件
            "font_path": None,           # 自定义字体路径，用于中文显示
            "use_gpu": True,             # 是否使用GPU加速
            "device": None,              # 设备: cuda:0, cpu
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        
        # 设置设备
        if self.config["device"] is None:
            if self.config["use_gpu"] and torch.cuda.is_available():
                self.config["device"] = "cuda:0"
            else:
                self.config["device"] = "cpu"
        
        logger.info(f"初始化SubtitleGenerator: {input_path} -> {self.output_path}")
        logger.info(f"配置: {self.config}")
        
        # 加载视频
        self.video = None
        self.duration = 0
        self.fps = 0
        self.load_video()
        
        # 加载Whisper模型
        self.model = None
    
    def load_video(self):
        """加载视频文件"""
        logger.info(f"加载视频: {self.input_path}")
        
        try:
            # 使用低内存模式加载视频
            self.video = VideoFileClip(self.input_path, audio=True, verbose=False)
            self.duration = self.video.duration
            self.fps = self.video.fps
            
            logger.info(f"视频时长: {self.duration:.2f}秒, FPS: {self.fps}")
        except Exception as e:
            logger.error(f"加载视频失败: {e}")
            raise
    
    def extract_audio(self):
        """从视频中提取音频"""
        logger.info("提取音频...")
        
        audio_temp = os.path.join(self.config["temp_dir"], "audio.wav")
        print(f"提取音频到临时文件... / Extracting audio to temp file...")
        
        # 使用较低的音频质量以减少内存使用
        self.video.audio.write_audiofile(
            audio_temp, 
            fps=16000,  # Whisper推荐的采样率
            nbytes=2,   # 16位音频
            codec='pcm_s16le',
            logger=None
        )
        
        return audio_temp
    
    def load_model(self):
        """加载Whisper语音识别模型"""
        logger.info(f"加载Whisper模型 ({self.config['model_size']})...")
        
        try:
            self.model = whisper.load_model(
                self.config["model_size"],
                device=self.config["device"]
            )
            logger.info(f"模型加载完成，使用设备: {self.config['device']}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        转录音频为字幕
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List: 字幕片段列表，每个元素为 (开始时间, 结束时间, 文本)
        """
        if self.model is None:
            self.load_model()
        
        logger.info("开始转录音频...")
        
        # 设置语言和任务类型
        language = None if self.config["language"] == "auto" else self.config["language"]
        
        # 转录音频
        try:
            transcribe_options = {
                "language": language,
                "task": "transcribe",
                "verbose": True
            }
            
            # 检测是否为中文
            if language == "zh" or (language is None and self._detect_chinese()):
                logger.info("检测到中文内容，使用中文转录设置")
                # 对于中文，我们可以添加一些特定设置
                transcribe_options["initial_prompt"] = "这是一个中文视频"
            
            result = self.model.transcribe(audio_path, **transcribe_options)
            
            # 提取字幕片段
            segments = []
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                if text:  # 只添加非空文本
                    segments.append((start, end, text))
            
            logger.info(f"转录完成，共 {len(segments)} 个字幕片段")
            return segments
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise
    
    def _detect_chinese(self):
        """检测视频是否包含中文内容"""
        # 这里可以使用简单的启发式方法，例如检测视频文件名或路径中是否包含中文字符
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        if chinese_pattern.search(self.input_path):
            return True
        
        # 也可以尝试对视频的前几秒进行采样分析
        return False
    
    def format_subtitles(self, segments):
        """
        格式化字幕片段
        
        Args:
            segments: 字幕片段列表
            
        Returns:
            List: 格式化后的字幕片段列表
        """
        logger.info("格式化字幕...")
        
        max_line_width = self.config["max_line_width"]
        formatted_segments = []
        
        for start, end, text in segments:
            # 分割长文本
            if len(text) > max_line_width:
                # 尝试在标点符号处分割
                lines = []
                current_line = ""
                
                # 中文标点符号和空格
                punctuation = "，。！？；：""''（）【】《》、,.!?;:\"'()[]<> "
                
                for char in text:
                    current_line += char
                    
                    # 如果当前行达到最大长度且下一个字符是标点符号，或者当前字符是标点符号
                    if len(current_line) >= max_line_width and (char in punctuation):
                        lines.append(current_line)
                        current_line = ""
                
                # 添加最后一行
                if current_line:
                    lines.append(current_line)
                
                formatted_text = "\n".join(lines)
            else:
                formatted_text = text
            
            formatted_segments.append((start, end, formatted_text))
        
        return formatted_segments
    
    def save_srt(self, segments, output_path=None):
        """
        保存SRT格式字幕文件
        
        Args:
            segments: 字幕片段列表
            output_path: 输出文件路径，默认为与输出视频同名的.srt文件
        """
        if output_path is None:
            output_path = os.path.splitext(self.output_path)[0] + ".srt"
        
        logger.info(f"保存SRT字幕文件: {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (start, end, text) in enumerate(segments, 1):
                # 转换时间格式为 HH:MM:SS,mmm
                start_str = self._format_time(start)
                end_str = self._format_time(end)
                
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")
    
    def _format_time(self, seconds):
        """将秒数转换为SRT时间格式 HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def create_subtitled_video(self, segments):
        """
        创建带字幕的视频
        
        Args:
            segments: 格式化后的字幕片段列表
        """
        logger.info("创建带字幕的视频...")
        
        # 获取配置
        font_size = self.config["font_size"]
        font_color = self.config["font_color"]
        stroke_color = self.config["stroke_color"]
        stroke_width = self.config["stroke_width"]
        position = self.config["position"]
        align = self.config["align"]
        font_path = self.config["font_path"]
        
        # 设置字体
        if font_path is None:
            # 尝试查找系统中的中文字体
            system_fonts = []
            if os.name == 'nt':  # Windows
                font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
                chinese_fonts = ['simhei.ttf', 'msyh.ttc', 'simsun.ttc']
                for font in chinese_fonts:
                    if os.path.exists(os.path.join(font_dir, font)):
                        font_path = os.path.join(font_dir, font)
                        break
            elif os.name == 'posix':  # Linux/Mac
                font_dirs = [
                    '/usr/share/fonts/',
                    '/usr/local/share/fonts/',
                    os.path.expanduser('~/.fonts/')
                ]
                chinese_fonts = ['wqy-microhei.ttc', 'wqy-zenhei.ttc', 'noto-cjk.ttc']
                for directory in font_dirs:
                    if os.path.exists(directory):
                        for root, _, files in os.walk(directory):
                            for font in chinese_fonts:
                                if font in files:
                                    font_path = os.path.join(root, font)
                                    break
            
            if font_path is None:
                logger.warning("未找到适合的中文字体，将使用默认字体")
        
        # 创建字幕剪辑
        subtitle_clips = []
        
        # 计算字幕位置
        video_width = self.video.w
        video_height = self.video.h
        
        if position == "bottom":
            y_position = video_height * 0.85  # 底部位置
        else:  # top
            y_position = video_height * 0.1   # 顶部位置
        
        for start, end, text in tqdm(segments, desc="生成字幕"):
            # 创建文本剪辑
            try:
                text_clip = TextClip(
                    text,
                    fontsize=font_size,
                    color=font_color,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                    align=align,
                    font=font_path,
                    method='caption'  # 使用caption方法支持多行文本
                )
                
                # 设置位置
                text_clip = text_clip.set_position(('center', y_position))
                
                # 设置时间
                text_clip = text_clip.set_start(start).set_end(end)
                
                subtitle_clips.append(text_clip)
            except Exception as e:
                logger.error(f"创建字幕剪辑失败: {e}")
                # 继续处理其他字幕
                continue
        
        # 合成视频
        logger.info("合成最终视频...")
        final_video = CompositeVideoClip([self.video] + subtitle_clips)
        
        # 导出视频
        logger.info(f"导出视频: {self.output_path}")
        final_video.write_videofile(
            self.output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(self.config["temp_dir"], "temp_audio.m4a"),
            remove_temp=True,
            fps=self.fps
        )
        
        # 关闭视频对象
        final_video.close()
    
    def cleanup(self):
        """清理临时文件"""
        logger.info("清理临时文件...")
        
        # 关闭视频对象
        if self.video is not None:
            self.video.close()
        
        # 删除临时目录
        if os.path.exists(self.config["temp_dir"]):
            shutil.rmtree(self.config["temp_dir"])
    
    def generate_subtitles(self):
        """
        生成字幕并创建带字幕的视频
        
        Returns:
            str: 输出视频路径
        """
        try:
            # 1. 提取音频
            audio_path = self.extract_audio()
            
            # 2. 转录音频
            segments = self.transcribe_audio(audio_path)
            
            # 3. 格式化字幕
            formatted_segments = self.format_subtitles(segments)
            
            # 4. 保存SRT文件（如果需要）
            if self.config["srt_output"]:
                self.save_srt(formatted_segments)
            
            # 5. 创建带字幕的视频
            self.create_subtitled_video(formatted_segments)
            
            logger.info(f"字幕生成完成: {self.output_path}")
            return self.output_path
            
        except Exception as e:
            logger.error(f"字幕生成失败: {e}")
            raise
        finally:
            self.cleanup()


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SubtitleGenerator - 为视频自动生成字幕")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("-o", "--output", help="输出视频文件路径")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                        default="medium", help="Whisper模型大小")
    parser.add_argument("--language", default="auto", help="语言代码，例如zh、en，auto为自动检测")
    parser.add_argument("--font-size", type=int, default=24, help="字幕字体大小")
    parser.add_argument("--font-color", default="white", help="字幕颜色")
    parser.add_argument("--stroke-color", default="black", help="字幕描边颜色")
    parser.add_argument("--stroke-width", type=float, default=1.5, help="字幕描边宽度")
    parser.add_argument("--position", choices=["bottom", "top"], default="bottom", help="字幕位置")
    parser.add_argument("--align", choices=["center", "left", "right"], default="center", help="字幕对齐方式")
    parser.add_argument("--max-line-width", type=int, default=40, help="每行最大字符数")
    parser.add_argument("--no-srt", action="store_false", dest="srt_output", help="不输出SRT文件")
    parser.add_argument("--font-path", help="自定义字体路径，用于中文显示")
    parser.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="不使用GPU加速")
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        "model_size": args.model,
        "language": args.language,
        "font_size": args.font_size,
        "font_color": args.font_color,
        "stroke_color": args.stroke_color,
        "stroke_width": args.stroke_width,
        "position": args.position,
        "align": args.align,
        "max_line_width": args.max_line_width,
        "srt_output": args.srt_output,
        "font_path": args.font_path,
        "use_gpu": args.use_gpu,
    }
    
    # 创建并运行SubtitleGenerator
    try:
        subtitle_generator = SubtitleGenerator(args.input, args.output, config)
        output_path = subtitle_generator.generate_subtitles()
        print(f"处理完成! 输出文件: {output_path}")
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
