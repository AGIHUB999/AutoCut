#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoCut - 自动剪辑演唱会高潮部分的AI工具
"""

import os
import sys
import time
import argparse
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import json
import tempfile
import shutil

# 尝试导入必要的库，如果不存在则提示安装
try:
    import cv2
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import torch
except ImportError as e:
    print(f"缺少必要的库: {e}")
    print("请运行: pip install opencv-python librosa matplotlib moviepy pydub torch numpy")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoCut")

class AutoCut:
    """自动剪辑演唱会高潮部分的主类"""
    
    def __init__(self, input_path: str, output_path: str, config: Dict = None):
        """
        初始化AutoCut
        
        Args:
            input_path: 输入视频文件路径
            output_path: 输出视频文件路径
            config: 配置参数
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # 默认配置
        self.config = {
            "min_clip_duration": 5,  # 最小片段时长(秒)
            "max_clip_duration": 30,  # 最大片段时长(秒)
            "energy_threshold": 0.7,  # 能量阈值(0-1)
            "applause_detection": True,  # 是否检测掌声
            "tempo_change_detection": True,  # 是否检测节奏变化
            "highlight_count": 5,  # 要提取的高潮片段数量
            "temp_dir": tempfile.mkdtemp(),  # 临时目录
            "fade_duration": 0.5,  # 淡入淡出时长(秒)
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        logger.info(f"初始化AutoCut: {input_path} -> {output_path}")
        logger.info(f"配置: {self.config}")
        
        # 加载视频
        self.video = None
        self.audio = None
        self.duration = 0
        self.fps = 0
        self.load_video()
    
    def load_video(self):
        """加载视频文件并提取音频"""
        logger.info(f"加载视频: {self.input_path}")
        
        try:
            self.video = VideoFileClip(self.input_path)
            self.duration = self.video.duration
            self.fps = self.video.fps
            
            # 提取音频到临时文件
            audio_temp = os.path.join(self.config["temp_dir"], "audio.wav")
            self.video.audio.write_audiofile(audio_temp, logger=None)
            self.audio = audio_temp
            
            logger.info(f"视频时长: {self.duration:.2f}秒, FPS: {self.fps}")
        except Exception as e:
            logger.error(f"加载视频失败: {e}")
            raise
    
    def analyze_audio(self) -> Dict:
        """
        分析音频特征
        
        Returns:
            Dict: 包含音频分析结果的字典
        """
        logger.info("分析音频特征...")
        
        # 加载音频
        y, sr = librosa.load(self.audio, sr=None)
        
        # 计算音频特征
        results = {}
        
        # 1. 能量/音量 (RMS)
        rms = librosa.feature.rms(y=y)[0]
        results["rms"] = rms / np.max(rms)  # 归一化
        
        # 2. 节奏/速度变化
        if self.config["tempo_change_detection"]:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            results["tempo"] = tempo
            results["beats"] = librosa.frames_to_time(beats, sr=sr)
            results["onset_strength"] = onset_env / np.max(onset_env)  # 归一化
        
        # 3. 掌声检测
        if self.config["applause_detection"]:
            # 掌声通常在2-8kHz范围内有较高能量
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # 简单的掌声检测启发式方法
            applause_score = np.zeros_like(rms)
            for i in range(len(rms)):
                frame_idx = min(i, len(spectral_centroid) - 1)
                if 2000 < spectral_centroid[frame_idx] < 8000 and spectral_bandwidth[frame_idx] > 1500:
                    applause_score[i] = rms[i]
            
            results["applause_score"] = applause_score / np.max(applause_score) if np.max(applause_score) > 0 else applause_score
        
        # 4. 频谱对比度 (可以帮助检测高潮部分)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=0)
        results["spectral_contrast"] = spectral_contrast / np.max(spectral_contrast)
        
        # 将时间帧转换为秒
        frames = np.arange(len(results["rms"]))
        results["times"] = librosa.frames_to_time(frames, sr=sr)
        
        return results
    
    def detect_highlights(self, audio_features: Dict) -> List[Tuple[float, float, float]]:
        """
        检测视频中的高潮部分
        
        Args:
            audio_features: 音频特征分析结果
            
        Returns:
            List[Tuple[float, float, float]]: 高潮片段列表，每个元素为 (开始时间, 结束时间, 分数)
        """
        logger.info("检测高潮部分...")
        
        times = audio_features["times"]
        rms = audio_features["rms"]
        
        # 组合特征计算综合分数
        combined_score = rms.copy()
        
        # 添加掌声分数
        if self.config["applause_detection"] and "applause_score" in audio_features:
            combined_score = combined_score * 0.7 + audio_features["applause_score"] * 0.3
        
        # 添加频谱对比度
        if "spectral_contrast" in audio_features:
            # 确保长度一致
            spec_contrast = np.interp(
                np.linspace(0, 1, len(combined_score)),
                np.linspace(0, 1, len(audio_features["spectral_contrast"])),
                audio_features["spectral_contrast"]
            )
            combined_score = combined_score * 0.8 + spec_contrast * 0.2
        
        # 添加节奏变化
        if self.config["tempo_change_detection"] and "onset_strength" in audio_features:
            onset_strength = np.interp(
                np.linspace(0, 1, len(combined_score)),
                np.linspace(0, 1, len(audio_features["onset_strength"])),
                audio_features["onset_strength"]
            )
            combined_score = combined_score * 0.7 + onset_strength * 0.3
        
        # 平滑分数曲线
        window_size = int(sr / hop_length * 2)  # 约2秒的窗口
        smoothed_score = np.convolve(combined_score, np.ones(window_size)/window_size, mode='same')
        
        # 查找局部最大值作为潜在高潮点
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smoothed_score, height=self.config["energy_threshold"], distance=window_size)
        
        # 转换峰值索引为时间
        peak_times = [times[p] for p in peaks if p < len(times)]
        
        # 根据峰值创建片段
        segments = []
        min_duration = self.config["min_clip_duration"]
        max_duration = self.config["max_clip_duration"]
        
        for peak_time in peak_times:
            # 找到峰值附近的最佳片段
            peak_idx = np.argmin(np.abs(times - peak_time))
            
            # 向前后扩展，直到分数低于阈值或达到最大时长
            start_idx = peak_idx
            while start_idx > 0 and times[peak_idx] - times[start_idx] < max_duration / 2:
                if smoothed_score[start_idx] < self.config["energy_threshold"] * 0.5:
                    break
                start_idx -= 1
            
            end_idx = peak_idx
            while end_idx < len(times) - 1 and times[end_idx] - times[peak_idx] < max_duration / 2:
                if smoothed_score[end_idx] < self.config["energy_threshold"] * 0.5:
                    break
                end_idx += 1
            
            start_time = max(0, times[start_idx])
            end_time = min(self.duration, times[end_idx])
            
            # 确保片段时长符合要求
            if end_time - start_time < min_duration:
                # 扩展到最小时长
                pad = (min_duration - (end_time - start_time)) / 2
                start_time = max(0, start_time - pad)
                end_time = min(self.duration, end_time + pad)
            
            # 计算片段平均分数
            segment_score = np.mean(smoothed_score[start_idx:end_idx+1])
            
            segments.append((start_time, end_time, segment_score))
        
        # 合并重叠片段
        segments = self._merge_overlapping_segments(segments)
        
        # 按分数排序并选择前N个
        segments.sort(key=lambda x: x[2], reverse=True)
        highlight_count = min(self.config["highlight_count"], len(segments))
        top_segments = segments[:highlight_count]
        
        # 按时间顺序排序
        top_segments.sort(key=lambda x: x[0])
        
        logger.info(f"检测到 {len(top_segments)} 个高潮片段")
        for i, (start, end, score) in enumerate(top_segments):
            logger.info(f"片段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s, 分数: {score:.2f})")
        
        return top_segments
    
    def _merge_overlapping_segments(self, segments: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """合并重叠的片段"""
        if not segments:
            return []
        
        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x[0])
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            previous = merged[-1]
            
            # 如果当前片段与上一个片段重叠
            if current[0] <= previous[1]:
                # 合并片段，取较高的分数
                merged[-1] = (
                    previous[0],
                    max(previous[1], current[1]),
                    max(previous[2], current[2])
                )
            else:
                merged.append(current)
        
        return merged
    
    def create_highlight_video(self, segments: List[Tuple[float, float, float]]) -> None:
        """
        根据检测到的高潮片段创建精彩集锦视频
        
        Args:
            segments: 高潮片段列表，每个元素为 (开始时间, 结束时间, 分数)
        """
        logger.info("创建精彩集锦视频...")
        
        if not segments:
            logger.warning("没有检测到高潮片段")
            return
        
        # 提取片段
        clips = []
        for i, (start, end, _) in enumerate(segments):
            logger.info(f"处理片段 {i+1}: {start:.2f}s - {end:.2f}s")
            
            # 提取片段并添加淡入淡出效果
            clip = self.video.subclip(start, end)
            clip = clip.fadein(self.config["fade_duration"]).fadeout(self.config["fade_duration"])
            clips.append(clip)
        
        # 连接所有片段
        final_clip = concatenate_videoclips(clips)
        
        # 写入输出文件
        logger.info(f"导出视频到: {self.output_path}")
        final_clip.write_videofile(
            self.output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(self.config["temp_dir"], "temp_audio.m4a"),
            remove_temp=True,
            threads=4,
            logger=None
        )
        
        logger.info(f"精彩集锦视频已保存到: {self.output_path}")
    
    def visualize_analysis(self, audio_features: Dict, segments: List[Tuple[float, float, float]], output_path: str) -> None:
        """
        可视化音频分析和高潮片段
        
        Args:
            audio_features: 音频特征分析结果
            segments: 高潮片段列表
            output_path: 输出图像路径
        """
        logger.info(f"生成分析可视化图表: {output_path}")
        
        plt.figure(figsize=(15, 10))
        
        # 绘制RMS能量
        plt.subplot(3, 1, 1)
        plt.plot(audio_features["times"], audio_features["rms"])
        plt.title("音频能量 (RMS)")
        plt.xlabel("时间 (秒)")
        plt.ylabel("归一化能量")
        
        # 标记高潮片段
        for start, end, score in segments:
            plt.axvspan(start, end, alpha=0.3, color='red')
        
        # 绘制频谱对比度
        if "spectral_contrast" in audio_features:
            plt.subplot(3, 1, 2)
            plt.plot(np.linspace(0, self.duration, len(audio_features["spectral_contrast"])), 
                    audio_features["spectral_contrast"])
            plt.title("频谱对比度")
            plt.xlabel("时间 (秒)")
            plt.ylabel("归一化对比度")
            
            # 标记高潮片段
            for start, end, score in segments:
                plt.axvspan(start, end, alpha=0.3, color='red')
        
        # 绘制掌声分数
        if "applause_score" in audio_features:
            plt.subplot(3, 1, 3)
            plt.plot(audio_features["times"], audio_features["applause_score"])
            plt.title("掌声检测分数")
            plt.xlabel("时间 (秒)")
            plt.ylabel("归一化分数")
            
            # 标记高潮片段
            for start, end, score in segments:
                plt.axvspan(start, end, alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"分析可视化已保存到: {output_path}")
    
    def cleanup(self):
        """清理临时文件"""
        logger.info("清理临时文件...")
        
        # 关闭视频对象
        if self.video is not None:
            self.video.close()
        
        # 删除临时目录
        if os.path.exists(self.config["temp_dir"]):
            shutil.rmtree(self.config["temp_dir"])
    
    def process(self) -> str:
        """
        处理视频并生成精彩集锦
        
        Returns:
            str: 输出视频路径
        """
        try:
            # 1. 分析音频
            audio_features = self.analyze_audio()
            
            # 2. 检测高潮部分
            segments = self.detect_highlights(audio_features)
            
            # 3. 创建精彩集锦视频
            self.create_highlight_video(segments)
            
            # 4. 可视化分析结果
            vis_path = os.path.splitext(self.output_path)[0] + "_analysis.png"
            self.visualize_analysis(audio_features, segments, vis_path)
            
            return self.output_path
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            raise
        finally:
            self.cleanup()


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="AutoCut - 自动剪辑演唱会高潮部分的AI工具")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", help="输出视频文件路径")
    parser.add_argument("--min-duration", type=float, default=5, help="最小片段时长(秒)")
    parser.add_argument("--max-duration", type=float, default=30, help="最大片段时长(秒)")
    parser.add_argument("--threshold", type=float, default=0.7, help="能量阈值(0-1)")
    parser.add_argument("--count", type=int, default=5, help="要提取的高潮片段数量")
    parser.add_argument("--no-applause", action="store_false", dest="applause", help="禁用掌声检测")
    parser.add_argument("--no-tempo", action="store_false", dest="tempo", help="禁用节奏变化检测")
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        "min_clip_duration": args.min_duration,
        "max_clip_duration": args.max_duration,
        "energy_threshold": args.threshold,
        "applause_detection": args.applause,
        "tempo_change_detection": args.tempo,
        "highlight_count": args.count,
    }
    
    # 创建并运行AutoCut
    try:
        autocut = AutoCut(args.input, args.output, config)
        output_path = autocut.process()
        print(f"处理完成! 输出文件: {output_path}")
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    # 设置librosa缓存
    os.environ['LIBROSA_CACHE_DIR'] = tempfile.mkdtemp()
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义全局变量
    sr = 22050  # 采样率
    hop_length = 512  # 帧跳跃长度
    
    sys.exit(main())
