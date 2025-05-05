#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vertical Converter - 将横屏视频转换为适合短视频平台的竖屏格式
"""

import os
import sys
import argparse
import subprocess
import json
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VerticalConverter")

class VerticalConverter:
    """将横屏视频转换为竖屏格式的工具"""
    
    def __init__(self, input_path, output_path, **kwargs):
        """
        初始化VerticalConverter
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径
            **kwargs: 其他配置参数
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # 默认配置
        self.config = {
            "output_width": 1080,        # 输出视频宽度
            "output_height": 1920,       # 输出视频高度
            "focus_mode": "auto",        # 焦点模式: auto, center, face, motion
            "background_blur": 30,       # 背景模糊程度 (0-100)
            "background_color": "black", # 背景颜色
            "zoom_factor": 1.2,          # 放大因子
            "temp_dir": tempfile.mkdtemp(),  # 临时目录
            "quality": "medium",         # 质量: low, medium, high
            "fps": None,                 # 帧率，None表示保持原帧率
            "audio_bitrate": "192k",     # 音频比特率
            "add_caption": False,        # 是否添加标题
            "caption_text": "",          # 标题文本
            "use_gpu": False,            # 是否使用GPU加速
        }
        
        # 更新配置
        self.config.update(kwargs)
        
        # 检查FFmpeg是否可用
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg未安装或不可用。请安装FFmpeg并确保它在系统PATH中。")
            raise RuntimeError("FFmpeg未安装或不可用")
        
        # 检查是否安装了OpenCV（用于人脸检测和运动检测）
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            logger.warning("未安装OpenCV，将无法使用人脸检测和运动检测功能。")
            self.has_cv2 = False
        
        logger.info(f"初始化VerticalConverter: {input_path} -> {output_path}")
        logger.info(f"配置: {self.config}")
        
        # 初始化视频信息
        self.video_info = None
        self.width = None
        self.height = None
        self.duration = None
        self.fps = None
    
    def __del__(self):
        """清理临时文件"""
        try:
            if hasattr(self, 'config') and 'temp_dir' in self.config:
                shutil.rmtree(self.config["temp_dir"], ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")
    
    def get_video_info(self):
        """获取视频信息"""
        print("获取视频信息... / Getting video information...")
        
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration:stream=width,height,r_frame_rate,codec_name", 
            "-of", "json", 
            self.input_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(result.stdout)
            
            # 提取信息
            self.duration = float(info["format"]["duration"])
            
            # 找到视频流
            video_stream = None
            for stream in info["streams"]:
                if "width" in stream and "height" in stream:
                    video_stream = stream
                    break
            
            if video_stream:
                self.width = video_stream["width"]
                self.height = video_stream["height"]
                
                # 解析帧率（通常是分数形式，如"24000/1001"）
                fps_parts = video_stream["r_frame_rate"].split('/')
                if len(fps_parts) == 2:
                    self.fps = float(fps_parts[0]) / float(fps_parts[1])
                else:
                    self.fps = float(fps_parts[0])
                
                # 获取编解码器
                self.codec = video_stream.get("codec_name", "h264")
            
            self.video_info = info
            print(f"视频信息: {self.width}x{self.height}, {self.fps:.2f} FPS, {self.duration:.2f}秒, 编解码器: {self.codec}")
            print(f"Video info: {self.width}x{self.height}, {self.fps:.2f} FPS, {self.duration:.2f}s, Codec: {self.codec}")
            
            # 计算宽高比
            self.aspect_ratio = self.width / self.height
            print(f"宽高比: {self.aspect_ratio:.2f}")
            print(f"Aspect ratio: {self.aspect_ratio:.2f}")
            
            return info
        
        except subprocess.SubprocessError as e:
            logger.error(f"获取视频信息失败: {e}")
            raise
    
    def detect_faces(self, frame_count=10):
        """
        检测视频中的人脸位置
        
        Args:
            frame_count: 要分析的帧数
        
        Returns:
            List[Tuple[int, int, int, int]]: 人脸位置列表 (x, y, w, h)
        """
        if not self.has_cv2:
            logger.warning("未安装OpenCV，无法使用人脸检测功能。")
            return []
        
        print("检测人脸位置... / Detecting face positions...")
        
        # 加载人脸检测器
        face_cascade = self.cv2.CascadeClassifier(self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 打开视频
        cap = self.cv2.VideoCapture(self.input_path)
        
        # 计算要跳过的帧数
        total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // frame_count)
        
        faces = []
        
        for i in range(frame_count):
            # 跳到指定帧
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为灰度图
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            detected_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # 添加到结果列表
            for (x, y, w, h) in detected_faces:
                faces.append((x, y, w, h))
        
        cap.release()
        
        if faces:
            print(f"检测到 {len(faces)} 个人脸")
            print(f"Detected {len(faces)} faces")
        else:
            print("未检测到人脸 / No faces detected")
        
        return faces
    
    def detect_motion_areas(self, frame_count=20):
        """
        检测视频中的运动区域
        
        Args:
            frame_count: 要分析的帧数
        
        Returns:
            List[Tuple[int, int, int, int]]: 运动区域列表 (x, y, w, h)
        """
        if not self.has_cv2:
            logger.warning("未安装OpenCV，无法使用运动检测功能。")
            return []
        
        print("检测运动区域... / Detecting motion areas...")
        
        # 打开视频
        cap = self.cv2.VideoCapture(self.input_path)
        
        # 计算要跳过的帧数
        total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = max(1, total_frames // frame_count)
        
        # 初始化运动检测器
        fgbg = self.cv2.createBackgroundSubtractorMOG2()
        
        motion_areas = []
        
        prev_frame = None
        
        for i in range(frame_count):
            # 跳到指定帧
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用运动检测
            fgmask = fgbg.apply(frame)
            
            # 查找轮廓
            contours, _ = self.cv2.findContours(fgmask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤小轮廓
            min_area = (self.width * self.height) * 0.01  # 至少占总面积的1%
            significant_contours = [cnt for cnt in contours if self.cv2.contourArea(cnt) > min_area]
            
            # 为每个显著轮廓添加边界框
            for cnt in significant_contours:
                x, y, w, h = self.cv2.boundingRect(cnt)
                motion_areas.append((x, y, w, h))
        
        cap.release()
        
        if motion_areas:
            print(f"检测到 {len(motion_areas)} 个运动区域")
            print(f"Detected {len(motion_areas)} motion areas")
        else:
            print("未检测到显著运动 / No significant motion detected")
        
        return motion_areas
    
    def determine_focus_area(self):
        """
        确定视频的焦点区域
        
        Returns:
            Tuple[int, int, int, int]: 焦点区域 (x, y, w, h)
        """
        focus_mode = self.config["focus_mode"]
        
        # 默认焦点区域（居中）
        center_x = self.width // 2
        center_y = self.height // 2
        focus_width = int(self.height * (9/16))  # 假设目标是9:16的竖屏比例
        focus_height = self.height
        
        focus_x = max(0, center_x - focus_width // 2)
        focus_y = 0
        
        # 根据不同模式调整焦点区域
        if focus_mode == "center":
            # 已经是居中模式，不需要调整
            pass
        
        elif focus_mode == "face" and self.has_cv2:
            # 人脸检测模式
            faces = self.detect_faces()
            
            if faces:
                # 计算所有人脸的平均位置
                avg_x = sum(x + w//2 for x, y, w, h in faces) // len(faces)
                
                # 调整焦点区域，使人脸居中
                focus_x = max(0, min(self.width - focus_width, avg_x - focus_width // 2))
        
        elif focus_mode == "motion" and self.has_cv2:
            # 运动检测模式
            motion_areas = self.detect_motion_areas()
            
            if motion_areas:
                # 计算所有运动区域的平均位置
                avg_x = sum(x + w//2 for x, y, w, h in motion_areas) // len(motion_areas)
                
                # 调整焦点区域，使运动区域居中
                focus_x = max(0, min(self.width - focus_width, avg_x - focus_width // 2))
        
        elif focus_mode == "auto":
            # 自动模式：先尝试人脸检测，如果没有人脸再尝试运动检测
            if self.has_cv2:
                faces = self.detect_faces()
                
                if faces:
                    # 计算所有人脸的平均位置
                    avg_x = sum(x + w//2 for x, y, w, h in faces) // len(faces)
                    
                    # 调整焦点区域，使人脸居中
                    focus_x = max(0, min(self.width - focus_width, avg_x - focus_width // 2))
                else:
                    # 没有检测到人脸，尝试运动检测
                    motion_areas = self.detect_motion_areas()
                    
                    if motion_areas:
                        # 计算所有运动区域的平均位置
                        avg_x = sum(x + w//2 for x, y, w, h in motion_areas) // len(motion_areas)
                        
                        # 调整焦点区域，使运动区域居中
                        focus_x = max(0, min(self.width - focus_width, avg_x - focus_width // 2))
        
        print(f"焦点区域: x={focus_x}, y={focus_y}, 宽={focus_width}, 高={focus_height}")
        print(f"Focus area: x={focus_x}, y={focus_y}, width={focus_width}, height={focus_height}")
        
        return (focus_x, focus_y, focus_width, focus_height)
    
    def convert_to_vertical(self):
        """将横屏视频转换为竖屏格式"""
        print("转换为竖屏格式... / Converting to vertical format...")
        
        # 获取视频信息
        if not self.video_info:
            self.get_video_info()
        
        # 确定焦点区域
        focus_x, focus_y, focus_width, focus_height = self.determine_focus_area()
        
        # 准备FFmpeg命令
        output_width = self.config["output_width"]
        output_height = self.config["output_height"]
        
        # 设置质量参数
        quality_presets = {
            "low": {"crf": "28", "preset": "veryfast", "audio_bitrate": "128k"},
            "medium": {"crf": "23", "preset": "medium", "audio_bitrate": "192k"},
            "high": {"crf": "18", "preset": "slow", "audio_bitrate": "256k"}
        }
        
        quality = quality_presets[self.config["quality"]]
        
        # 设置帧率
        fps_arg = []
        if self.config["fps"]:
            fps_arg = ["-r", str(self.config["fps"])]
        
        # 设置GPU加速
        hwaccel_args = []
        if self.config["use_gpu"]:
            # 尝试使用NVIDIA GPU加速
            hwaccel_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        
        # 创建复杂的滤镜图
        filter_complex = []
        
        # 1. 裁剪焦点区域
        filter_complex.append(f"[0:v]crop={focus_width}:{focus_height}:{focus_x}:{focus_y}[cropped]")
        
        # 2. 缩放到目标尺寸
        filter_complex.append(f"[cropped]scale={output_width}:{output_height}:force_original_aspect_ratio=decrease[scaled]")
        
        # 3. 填充黑边以达到目标尺寸
        filter_complex.append(f"[scaled]pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color={self.config['background_color']}[padded]")
        
        # 4. 如果需要模糊背景
        if self.config["background_blur"] > 0:
            # 创建模糊的背景
            filter_complex.append(f"[0:v]scale={output_width}:{output_height}:force_original_aspect_ratio=increase,crop={output_width}:{output_height},boxblur={self.config['background_blur']}[blurred]")
            
            # 将裁剪区域叠加在模糊背景上
            filter_complex.append(f"[blurred][padded]overlay=(W-w)/2:(H-h)/2[video]")
        else:
            filter_complex.append("[padded]copy[video]")
        
        # 5. 如果需要添加标题
        if self.config["add_caption"] and self.config["caption_text"]:
            filter_complex.append(f"[video]drawtext=text='{self.config['caption_text']}':fontcolor=white:fontsize=40:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-th-20[captioned]")
            final_output = "[captioned]"
        else:
            final_output = "[video]"
        
        # 构建完整的FFmpeg命令
        cmd = [
            "ffmpeg",
            "-i", self.input_path
        ] + hwaccel_args + [
            "-filter_complex", ";".join(filter_complex),
            "-map", final_output,
            "-map", "0:a",
            "-c:v", "libx264",
            "-crf", quality["crf"],
            "-preset", quality["preset"],
            "-c:a", "aac",
            "-b:a", self.config["audio_bitrate"] or quality["audio_bitrate"],
            "-movflags", "+faststart"
        ] + fps_arg + [
            "-y",
            self.output_path
        ]
        
        print("执行转换... / Executing conversion...")
        print(f"命令: {' '.join(cmd)}")
        
        try:
            # 执行FFmpeg命令
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            print(f"转换完成！输出文件: {self.output_path}")
            print(f"Conversion complete! Output file: {self.output_path}")
            
            return True
        
        except subprocess.SubprocessError as e:
            logger.error(f"转换失败: {e}")
            
            # 尝试使用更简单的命令
            print("尝试使用备用方法... / Trying alternative method...")
            
            simple_cmd = [
                "ffmpeg",
                "-i", self.input_path,
                "-vf", f"crop={focus_width}:{focus_height}:{focus_x}:{focus_y},scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color={self.config['background_color']}",
                "-c:v", "libx264",
                "-crf", quality["crf"],
                "-preset", "veryfast",
                "-c:a", "aac",
                "-b:a", quality["audio_bitrate"],
                "-y",
                self.output_path
            ]
            
            try:
                subprocess.run(simple_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                print(f"使用备用方法转换完成！输出文件: {self.output_path}")
                print(f"Conversion complete using alternative method! Output file: {self.output_path}")
                
                return True
            
            except subprocess.SubprocessError as e2:
                logger.error(f"备用方法也失败: {e2}")
                return False
    
    def process(self):
        """处理视频"""
        try:
            # 1. 获取视频信息
            self.get_video_info()
            
            # 2. 转换为竖屏格式
            success = self.convert_to_vertical()
            
            # 3. 清理
            self.__del__()
            
            return success
        
        except Exception as e:
            logger.error(f"处理视频时出错: {e}")
            # 确保清理临时文件
            self.__del__()
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VerticalConverter - 将横屏视频转换为竖屏格式")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", help="输出视频文件路径")
    parser.add_argument("--width", type=int, default=1080, help="输出视频宽度")
    parser.add_argument("--height", type=int, default=1920, help="输出视频高度")
    parser.add_argument("--focus", choices=["auto", "center", "face", "motion"], default="auto", help="焦点模式")
    parser.add_argument("--blur", type=int, default=30, help="背景模糊程度 (0-100)")
    parser.add_argument("--bg-color", default="black", help="背景颜色")
    parser.add_argument("--zoom", type=float, default=1.2, help="放大因子")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="输出质量")
    parser.add_argument("--fps", type=int, help="输出帧率")
    parser.add_argument("--caption", help="添加标题文本")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    
    args = parser.parse_args()
    
    # 创建VerticalConverter实例
    converter = VerticalConverter(
        args.input,
        args.output,
        output_width=args.width,
        output_height=args.height,
        focus_mode=args.focus,
        background_blur=args.blur,
        background_color=args.bg_color,
        zoom_factor=args.zoom,
        quality=args.quality,
        fps=args.fps,
        add_caption=bool(args.caption),
        caption_text=args.caption or "",
        use_gpu=args.gpu
    )
    
    # 处理视频
    try:
        success = converter.process()
        if success:
            print("处理完成！/ Processing complete!")
            return 0
        else:
            print("处理失败。/ Processing failed.")
            return 1
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
