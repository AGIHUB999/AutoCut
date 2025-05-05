#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoCut FFmpeg版本 - 使用FFmpeg处理大型视频文件的低内存实现
"""

import os
import sys
import time
import argparse
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import shutil
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoCut-FFmpeg")

class AutoCutFFmpeg:
    """使用FFmpeg的AutoCut实现，专为大型视频文件设计"""
    
    def __init__(self, input_path, output_path, **kwargs):
        """
        初始化AutoCut
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径
            **kwargs: 其他配置参数
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # 默认配置
        self.config = {
            "min_clip_duration": 5,      # 最小片段时长（秒）
            "max_clip_duration": 30,     # 最大片段时长（秒）
            "energy_threshold": 0.7,     # 能量阈值
            "highlight_count": 5,        # 高潮片段数量
            "fade_duration": 0.5,        # 淡入淡出时长（秒）
            "chunk_size": 300,           # 分块大小（秒）
            "temp_dir": tempfile.mkdtemp(),  # 临时目录
            "volume_threshold": 0.8,     # 音量阈值
            "scene_threshold": 0.3,      # 场景变化阈值
        }
        
        # 更新配置
        self.config.update(kwargs)
        
        # 检查FFmpeg是否可用
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg未安装或不可用。请安装FFmpeg并确保它在系统PATH中。")
            raise RuntimeError("FFmpeg未安装或不可用")
        
        logger.info(f"初始化AutoCut: {input_path} -> {output_path}")
        logger.info(f"配置: {self.config}")
        
        # 初始化视频信息
        self.duration = None
        self.fps = None
        self.audio_file = None
        self.video_info = None
    
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
            "-show_entries", "format=duration:stream=width,height,r_frame_rate", 
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
                # 解析帧率（通常是分数形式，如"24000/1001"）
                fps_parts = video_stream["r_frame_rate"].split('/')
                if len(fps_parts) == 2:
                    self.fps = float(fps_parts[0]) / float(fps_parts[1])
                else:
                    self.fps = float(fps_parts[0])
                
                self.width = video_stream["width"]
                self.height = video_stream["height"]
            
            self.video_info = info
            print(f"视频时长: {self.duration:.2f}秒, FPS: {self.fps:.2f}, 分辨率: {self.width}x{self.height}")
            print(f"Video duration: {self.duration:.2f}s, FPS: {self.fps:.2f}, Resolution: {self.width}x{self.height}")
            
            return info
        
        except subprocess.SubprocessError as e:
            logger.error(f"获取视频信息失败: {e}")
            raise
    
    def extract_audio(self):
        """提取音频到临时文件"""
        print("提取音频... / Extracting audio...")
        
        audio_file = os.path.join(self.config["temp_dir"], "audio.wav")
        
        cmd = [
            "ffmpeg",
            "-i", self.input_path,
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",  # 16位PCM
            "-ar", "22050",  # 采样率
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            audio_file
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.audio_file = audio_file
            print(f"音频已提取到: {audio_file}")
            print(f"Audio extracted to: {audio_file}")
            return audio_file
        
        except subprocess.SubprocessError as e:
            logger.error(f"提取音频失败: {e}")
            raise
    
    def analyze_audio_volume(self):
        """分析音频音量"""
        print("分析音频音量... / Analyzing audio volume...")
        
        if not self.audio_file:
            self.extract_audio()
        
        # 使用FFmpeg的loudnorm滤镜分析音量
        volume_data_file = os.path.join(self.config["temp_dir"], "volume_data.txt")
        
        # 分块处理
        chunk_size = self.config["chunk_size"]
        num_chunks = int(np.ceil(self.duration / chunk_size))
        print(f"将音频分为 {num_chunks} 个块进行处理 / Dividing audio into {num_chunks} chunks")
        
        # 初始化结果
        volume_data = []
        
        for i in range(num_chunks):
            start_time = i * chunk_size
            duration = min(chunk_size, self.duration - start_time)
            
            print(f"处理块 {i+1}/{num_chunks}: {start_time:.1f}s - {start_time+duration:.1f}s")
            print(f"Processing chunk {i+1}/{num_chunks}: {start_time:.1f}s - {start_time+duration:.1f}s")
            
            # 使用FFmpeg的volumedetect滤镜分析音量
            cmd = [
                "ffmpeg",
                "-i", self.audio_file,
                "-ss", str(start_time),
                "-t", str(duration),
                "-af", "volumedetect",
                "-f", "null",
                "-y",
                "NUL"  # Windows上使用NUL，Linux/Mac上使用/dev/null
            ]
            
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # 解析输出
                for line in result.stderr.split('\n'):
                    if "mean_volume" in line:
                        mean_vol = float(line.split(':')[1].strip().split(' ')[0])
                    elif "max_volume" in line:
                        max_vol = float(line.split(':')[1].strip().split(' ')[0])
                
                # 将dB值转换为线性值（0-1范围）
                # dB = 20 * log10(linear)
                # linear = 10^(dB/20)
                mean_linear = 10 ** (mean_vol / 20) if mean_vol < 0 else 1.0
                max_linear = 10 ** (max_vol / 20) if max_vol < 0 else 1.0
                
                # 为这个块的每一秒添加音量数据
                for j in range(int(duration)):
                    timestamp = start_time + j
                    volume_data.append({
                        "time": timestamp,
                        "mean_volume": mean_linear,
                        "max_volume": max_linear
                    })
            
            except subprocess.SubprocessError as e:
                logger.error(f"分析块 {i+1} 音量失败: {e}")
                # 继续处理下一个块
        
        # 保存音量数据
        with open(volume_data_file, 'w') as f:
            json.dump(volume_data, f)
        
        print(f"音量分析完成，数据保存到: {volume_data_file}")
        print(f"Volume analysis complete, data saved to: {volume_data_file}")
        
        return volume_data
    
    def detect_scene_changes(self):
        """检测场景变化"""
        print("检测场景变化... / Detecting scene changes...")
        
        scene_data_file = os.path.join(self.config["temp_dir"], "scene_data.txt")
        
        # 使用FFmpeg的scene检测滤镜
        cmd = [
            "ffmpeg",
            "-i", self.input_path,
            "-filter:v", f"select='gt(scene,{self.config['scene_threshold']})',showinfo",
            "-f", "null",
            "-y",
            "NUL"  # Windows上使用NUL，Linux/Mac上使用/dev/null
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 解析输出，查找场景变化
            scene_changes = []
            for line in result.stderr.split('\n'):
                if "pts_time" in line:
                    try:
                        time_str = line.split("pts_time:")[1].split()[0]
                        scene_changes.append(float(time_str))
                    except (IndexError, ValueError):
                        continue
            
            # 保存场景变化数据
            with open(scene_data_file, 'w') as f:
                json.dump(scene_changes, f)
            
            print(f"检测到 {len(scene_changes)} 个场景变化")
            print(f"Detected {len(scene_changes)} scene changes")
            
            return scene_changes
        
        except subprocess.SubprocessError as e:
            logger.error(f"检测场景变化失败: {e}")
            return []
    
    def detect_highlights(self):
        """检测视频高潮部分"""
        print("检测视频高潮部分... / Detecting video highlights...")
        
        # 获取音量数据
        volume_data = self.analyze_audio_volume()
        
        # 获取场景变化数据
        scene_changes = self.detect_scene_changes()
        
        # 结合音量和场景变化检测高潮
        highlights = []
        
        # 根据音量找到潜在的高潮点
        volume_threshold = self.config["volume_threshold"]
        max_volumes = [data["max_volume"] for data in volume_data]
        
        if max_volumes:
            # 归一化音量
            max_vol = max(max_volumes)
            if max_vol > 0:
                normalized_volumes = [v / max_vol for v in max_volumes]
            else:
                normalized_volumes = max_volumes
            
            # 找到超过阈值的点
            high_volume_points = []
            for i, vol in enumerate(normalized_volumes):
                if vol >= volume_threshold:
                    high_volume_points.append(i)
            
            # 将连续的高音量点合并为片段
            if high_volume_points:
                segments = []
                current_segment = [high_volume_points[0]]
                
                for i in range(1, len(high_volume_points)):
                    if high_volume_points[i] - high_volume_points[i-1] <= 5:  # 如果点之间间隔不超过5秒
                        current_segment.append(high_volume_points[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [high_volume_points[i]]
                
                segments.append(current_segment)
                
                # 转换为时间范围
                for segment in segments:
                    start_time = volume_data[segment[0]]["time"]
                    end_time = volume_data[segment[-1]]["time"] + 1  # +1秒确保包含最后一秒
                    
                    # 调整片段长度
                    duration = end_time - start_time
                    if duration < self.config["min_clip_duration"]:
                        # 如果片段太短，向两边扩展
                        extension = (self.config["min_clip_duration"] - duration) / 2
                        start_time = max(0, start_time - extension)
                        end_time = min(self.duration, end_time + extension)
                    elif duration > self.config["max_clip_duration"]:
                        # 如果片段太长，取中间部分
                        middle = (start_time + end_time) / 2
                        half_max = self.config["max_clip_duration"] / 2
                        start_time = middle - half_max
                        end_time = middle + half_max
                    
                    # 考虑场景变化
                    # 如果片段附近有场景变化，调整片段边界到场景变化点
                    for scene_time in scene_changes:
                        # 如果场景变化点在片段开始前5秒内，将片段开始调整到场景变化点
                        if 0 < start_time - scene_time < 5:
                            start_time = scene_time
                        # 如果场景变化点在片段结束后5秒内，将片段结束调整到场景变化点
                        elif 0 < scene_time - end_time < 5:
                            end_time = scene_time
                    
                    # 添加到高潮列表
                    highlights.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "score": np.mean([normalized_volumes[i] for i in segment])
                    })
        
        # 按分数排序
        highlights.sort(key=lambda x: x["score"], reverse=True)
        
        # 限制高潮数量
        highlights = highlights[:self.config["highlight_count"]]
        
        # 按时间顺序排序
        highlights.sort(key=lambda x: x["start"])
        
        print(f"检测到 {len(highlights)} 个高潮片段:")
        print(f"Detected {len(highlights)} highlight segments:")
        
        for i, h in enumerate(highlights):
            print(f"  {i+1}. {h['start']:.1f}s - {h['end']:.1f}s (时长: {h['duration']:.1f}s, 分数: {h['score']:.2f})")
            print(f"  {i+1}. {h['start']:.1f}s - {h['end']:.1f}s (Duration: {h['duration']:.1f}s, Score: {h['score']:.2f})")
        
        return highlights
    
    def create_highlight_video(self, highlights):
        """创建高潮视频"""
        print("创建高潮视频... / Creating highlight video...")
        
        if not highlights:
            logger.error("没有检测到高潮片段")
            return False
        
        # 为每个高潮片段创建一个临时文件
        temp_files = []
        
        for i, highlight in enumerate(highlights):
            temp_file = os.path.join(self.config["temp_dir"], f"highlight_{i}.mp4")
            
            # 使用更可靠的参数设置
            start_time = highlight["start"]
            duration = highlight["duration"]
            
            print(f"提取片段 {i+1}/{len(highlights)}: {start_time:.1f}s - {start_time+duration:.1f}s")
            print(f"Extracting segment {i+1}/{len(highlights)}: {start_time:.1f}s - {start_time+duration:.1f}s")
            
            try:
                # 使用更简单的命令行参数，提高兼容性
                cmd = [
                    "ffmpeg",
                    "-i", self.input_path,
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c", "copy",  # 直接复制流，不重新编码，速度更快且不会有质量损失
                    "-avoid_negative_ts", "1",
                    "-y",
                    temp_file
                ]
                
                # 首先尝试直接复制
                try:
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    temp_files.append(temp_file)
                    print(f"片段 {i+1} 提取成功 / Segment {i+1} extracted successfully")
                except subprocess.SubprocessError as e:
                    print(f"直接复制失败，尝试重新编码: {e}")
                    print(f"Direct copy failed, trying re-encoding: {e}")
                    
                    # 如果直接复制失败，尝试重新编码
                    cmd = [
                        "ffmpeg",
                        "-i", self.input_path,
                        "-ss", str(start_time),
                        "-t", str(duration),
                        "-c:v", "libx264",  # 使用H.264编码
                        "-c:a", "aac",      # 使用AAC音频编码
                        "-preset", "ultrafast",  # 最快的编码速度
                        "-crf", "23",       # 合理的质量
                        "-y",
                        temp_file
                    ]
                    
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    temp_files.append(temp_file)
                    print(f"片段 {i+1} 重新编码成功 / Segment {i+1} re-encoded successfully")
            
            except subprocess.SubprocessError as e:
                logger.error(f"提取片段 {i+1} 失败: {e}")
                print(f"尝试使用替代方法... / Trying alternative method...")
                
                # 如果上述方法都失败，尝试先提取到临时文件再处理
                try:
                    # 先将整个视频转换为更兼容的格式
                    temp_input = os.path.join(self.config["temp_dir"], "temp_input.mp4")
                    
                    # 使用最简单的命令提取片段
                    cmd = [
                        "ffmpeg",
                        "-i", self.input_path,
                        "-ss", str(max(0, start_time - 1)),  # 提前1秒开始
                        "-t", str(duration + 2),  # 多提取2秒
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-preset", "ultrafast",
                        "-y",
                        temp_file
                    ]
                    
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    temp_files.append(temp_file)
                    print(f"片段 {i+1} 使用替代方法提取成功 / Segment {i+1} extracted successfully using alternative method")
                
                except subprocess.SubprocessError as e2:
                    logger.error(f"所有提取方法都失败: {e2}")
                    print(f"无法提取此片段，跳过... / Cannot extract this segment, skipping...")
                    continue
        
        if not temp_files:
            logger.error("没有成功提取任何片段")
            
            # 尝试直接复制一个片段作为输出
            print("尝试直接提取一个片段作为输出... / Trying to extract a single segment as output...")
            
            try:
                # 选择视频中间的一个30秒片段
                middle_time = self.duration / 2
                
                cmd = [
                    "ffmpeg",
                    "-i", self.input_path,
                    "-ss", str(middle_time),
                    "-t", "30",
                    "-c", "copy",
                    "-y",
                    self.output_path
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                print(f"已创建单个片段视频: {self.output_path}")
                print(f"Created single segment video: {self.output_path}")
                return True
            
            except subprocess.SubprocessError as e:
                logger.error(f"直接提取失败: {e}")
                return False
        
        # 如果只有一个片段，直接复制为输出文件
        if len(temp_files) == 1:
            try:
                shutil.copy2(temp_files[0], self.output_path)
                print(f"高潮视频已创建: {self.output_path}")
                print(f"Highlight video created: {self.output_path}")
                return True
            except Exception as e:
                logger.error(f"复制文件失败: {e}")
                return False
        
        # 创建一个文件列表供FFmpeg使用
        concat_file = os.path.join(self.config["temp_dir"], "concat.txt")
        with open(concat_file, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")
        
        # 使用FFmpeg合并片段
        try:
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",  # 直接复制，不重新编码
                "-y",
                self.output_path
            ]
            
            print("合并片段... / Merging segments...")
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"高潮视频已创建: {self.output_path}")
            print(f"Highlight video created: {self.output_path}")
            return True
        
        except subprocess.SubprocessError as e:
            logger.error(f"合并片段失败: {e}")
            print("尝试使用替代方法合并... / Trying alternative merging method...")
            
            try:
                # 尝试使用重新编码的方式合并
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-preset", "ultrafast",
                    "-y",
                    self.output_path
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                print(f"高潮视频已创建(使用替代方法): {self.output_path}")
                print(f"Highlight video created (using alternative method): {self.output_path}")
                return True
            
            except subprocess.SubprocessError as e2:
                logger.error(f"所有合并方法都失败: {e2}")
                
                # 如果合并失败，至少保存第一个片段
                if temp_files:
                    try:
                        shutil.copy2(temp_files[0], self.output_path)
                        print(f"无法合并所有片段，已保存第一个片段: {self.output_path}")
                        print(f"Could not merge all segments, saved the first segment: {self.output_path}")
                        return True
                    except Exception as e3:
                        logger.error(f"保存第一个片段失败: {e3}")
                
                return False
    
    def process(self):
        """处理视频"""
        try:
            # 1. 获取视频信息
            self.get_video_info()
            
            # 2. 检测高潮
            highlights = self.detect_highlights()
            
            # 3. 创建高潮视频
            success = self.create_highlight_video(highlights)
            
            # 4. 清理
            self.__del__()
            
            return success
        
        except Exception as e:
            logger.error(f"处理视频时出错: {e}")
            # 确保清理临时文件
            self.__del__()
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AutoCut - 自动剪辑视频高潮部分")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", help="输出视频文件路径")
    parser.add_argument("--min-duration", type=float, default=5, help="最小片段时长（秒）")
    parser.add_argument("--max-duration", type=float, default=30, help="最大片段时长（秒）")
    parser.add_argument("--volume-threshold", type=float, default=0.8, help="音量阈值（0-1）")
    parser.add_argument("--scene-threshold", type=float, default=0.3, help="场景变化阈值（0-1）")
    parser.add_argument("--highlight-count", type=int, default=5, help="高潮片段数量")
    parser.add_argument("--chunk-size", type=int, default=300, help="分块大小（秒）")
    
    args = parser.parse_args()
    
    # 创建AutoCut实例
    autocut = AutoCutFFmpeg(
        args.input,
        args.output,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
        volume_threshold=args.volume_threshold,
        scene_threshold=args.scene_threshold,
        highlight_count=args.highlight_count,
        chunk_size=args.chunk_size
    )
    
    # 处理视频
    try:
        success = autocut.process()
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
