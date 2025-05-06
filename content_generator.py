#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Content Generator - 自动分析视频内容并生成吸引眼球的标题和封面
"""

import os
import sys
import argparse
import subprocess
import json
import tempfile
import shutil
import logging
import time
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Optional
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import textwrap
import random
import re
import cv2

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ContentGenerator")

class ContentGenerator:
    """分析视频内容并生成标题和封面的工具"""
    
    def __init__(self, input_path, output_dir=None, **kwargs):
        """
        初始化ContentGenerator
        
        Args:
            input_path: 输入视频文件路径
            output_dir: 输出目录，默认为视频文件所在目录
            **kwargs: 其他配置参数
        """
        self.input_path = input_path
        
        # 如果未指定输出目录，使用输入文件所在目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.dirname(os.path.abspath(input_path))
        
        # 确保输出目录存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"输出目录: {self.output_dir}")
        except Exception as e:
            print(f"创建输出目录时出错: {e}")
            # 尝试使用当前目录作为备选
            self.output_dir = os.path.abspath(".")
            print(f"使用当前目录作为备选: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 默认配置
        self.config = {
            "temp_dir": tempfile.mkdtemp(),  # 临时目录
            "frame_interval": 5,             # 截取帧的间隔（秒）
            "speech_recognition": True,      # 是否启用语音识别
            "ocr": True,                     # 是否启用OCR文本识别
            "title_style": "exciting",       # 标题风格: exciting, funny, emotional, informative
            "cover_style": "modern",         # 封面风格: modern, minimal, dramatic, colorful
            "language": "auto",              # 语言: auto, zh, en
            "max_title_length": 50,          # 标题最大长度
            "api_key": "",                   # API密钥
            "api_type": "",                  # API类型: openai, gemini
            "font_path": None,               # 字体路径
            "use_gpu": False,                # 是否使用GPU加速
            "thumbnail_count": 5,            # 生成的候选封面数量
            "title_count": 3,                # 生成的候选标题数量
        }
        
        # 更新配置
        self.config.update(kwargs)
        
        # 检查FFmpeg是否可用
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg未安装或不可用。请安装FFmpeg并确保它在系统PATH中。")
            raise RuntimeError("FFmpeg未安装或不可用")
        
        # 检查是否安装了OpenCV
        try:
            import cv2
            self.cv2 = cv2
            self.has_cv2 = True
        except ImportError:
            logger.warning("未安装OpenCV，将无法使用部分图像处理功能。")
            self.has_cv2 = False
        
        # 检查是否安装了语音识别库
        self.has_speech_recognition = False
        if self.config["speech_recognition"]:
            try:
                import speech_recognition as sr
                self.sr = sr
                self.has_speech_recognition = True
            except ImportError:
                logger.warning("未安装speech_recognition库，将无法使用语音识别功能。")
        
        # 检查是否安装了OCR库
        self.has_ocr = False
        if self.config["ocr"]:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                self.has_ocr = True
            except ImportError:
                logger.warning("未安装pytesseract库，将无法使用OCR文本识别功能。")
        
        # 检查是否有API密钥
        self.has_api = bool(self.config["api_key"])
        
        # 确定API类型
        if self.has_api:
            if self.config["api_key"].startswith("AIza"):
                self.config["api_type"] = "gemini"
                self.has_gemini = True
                logger.info("检测到Gemini API密钥")
            elif self.config["api_key"].startswith("sk-"):
                self.config["api_type"] = "openai"
                self.has_openai = True
                logger.info("检测到OpenAI API密钥")
            else:
                self.config["api_type"] = "unknown"
                logger.warning("无法识别的API密钥类型")
        else:
            self.has_gemini = False
            self.has_openai = False
        
        logger.info(f"初始化ContentGenerator: {input_path}")
        logger.info(f"配置: {self.config}")
        
        # 初始化视频信息
        self.video_info = None
        self.width = None
        self.height = None
        self.duration = None
        self.fps = None
        
        # 初始化结果
        self.transcription = ""
        self.ocr_text = ""
        self.key_frames = []
        self.titles = []
        self.thumbnails = []
    
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
            
            return info
        
        except subprocess.SubprocessError as e:
            logger.error(f"获取视频信息失败: {e}")
            raise

    def extract_audio(self):
        """从视频中提取音频"""
        print("提取音频... / Extracting audio...")
        
        audio_file = os.path.join(self.config["temp_dir"], "audio.wav")
        
        cmd = [
            "ffmpeg",
            "-i", self.input_path,
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",  # 16位PCM
            "-ar", "16000",  # 采样率
            "-ac", "1",  # 单声道
            "-y",  # 覆盖输出文件
            audio_file
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"音频已提取到: {audio_file}")
            print(f"Audio extracted to: {audio_file}")
            return audio_file
        
        except subprocess.SubprocessError as e:
            logger.error(f"提取音频失败: {e}")
            raise
    
    def extract_key_frames(self):
        """提取视频的关键帧"""
        print("提取关键帧... / Extracting key frames...")
        
        if not self.video_info:
            self.get_video_info()
        
        # 创建关键帧目录
        frames_dir = os.path.join(self.config["temp_dir"], "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 计算要提取的帧数
        interval = self.config["frame_interval"]
        num_frames = min(20, int(self.duration / interval))
        
        key_frames = []
        
        # 提取关键帧
        for i in range(num_frames):
            time_pos = i * interval
            frame_file = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            
            cmd = [
                "ffmpeg",
                "-ss", str(time_pos),
                "-i", self.input_path,
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                frame_file
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                key_frames.append({
                    "file": frame_file,
                    "time": time_pos
                })
            except subprocess.SubprocessError as e:
                logger.warning(f"提取帧 {i} 失败: {e}")
                continue
        
        print(f"提取了 {len(key_frames)} 个关键帧")
        print(f"Extracted {len(key_frames)} key frames")
        
        self.key_frames = key_frames
        return key_frames
    
    def recognize_speech(self):
        """使用语音识别从音频中提取文本"""
        if not self.has_speech_recognition:
            logger.warning("未安装speech_recognition库，无法使用语音识别功能。")
            print("未安装speech_recognition库，使用文件名作为文本内容...")
            filename = os.path.basename(self.input_path)
            filename = os.path.splitext(filename)[0]
            filename = filename.replace("_", " ").replace("-", " ")
            self.transcription = f"这是一个关于{filename}的视频"
            return self.transcription
        
        print("进行语音识别... / Performing speech recognition...")
        
        # 提取音频
        audio_file = self.extract_audio()
        
        # 使用语音识别
        recognizer = self.sr.Recognizer()
        
        # 对于较长的音频，分段处理
        segment_duration = 60  # 每段60秒
        num_segments = int(np.ceil(self.duration / segment_duration))
        
        transcription = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, self.duration)
            
            print(f"处理音频段 {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
            print(f"Processing audio segment {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
            
            # 提取音频段
            segment_file = os.path.join(self.config["temp_dir"], f"segment_{i:04d}.wav")
            
            cmd = [
                "ffmpeg",
                "-i", audio_file,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-y",
                segment_file
            ]
            
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # 识别音频段
                with self.sr.AudioFile(segment_file) as source:
                    audio_data = recognizer.record(source)
                    
                    try:
                        # 根据配置的语言选择识别引擎
                        if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()):
                            # 使用百度API或其他中文语音识别服务
                            if self.has_api:
                                if self.config["api_type"] == "gemini":
                                    text = self._recognize_with_gemini(audio_data)
                                elif self.config["api_type"] == "openai":
                                    text = self._recognize_with_openai(audio_data)
                                else:
                                    text = recognizer.recognize_google(audio_data, language="zh-CN")
                            else:
                                text = recognizer.recognize_google(audio_data, language="zh-CN")
                        else:
                            # 使用Google语音识别
                            text = recognizer.recognize_google(audio_data)
                        
                        transcription.append(text)
                    
                    except self.sr.UnknownValueError:
                        logger.warning(f"无法识别音频段 {i+1}")
                    except self.sr.RequestError as e:
                        logger.error(f"语音识别服务请求失败: {e}")
                
                # 删除临时文件
                os.remove(segment_file)
            
            except subprocess.SubprocessError as e:
                logger.warning(f"提取音频段 {i+1} 失败: {e}")
                continue
        
        # 合并转录文本
        full_transcription = " ".join(transcription)
        
        # 如果转录文本为空，使用文件名
        if not full_transcription.strip():
            print("语音识别未能提取有效文本，使用文件名作为备选")
            filename = os.path.basename(self.input_path)
            filename = os.path.splitext(filename)[0]
            filename = filename.replace("_", " ").replace("-", " ")
            full_transcription = f"这是一个关于{filename}的视频"
        
        print(f"语音识别完成，识别出 {len(full_transcription)} 个字符")
        print(f"Speech recognition complete, recognized {len(full_transcription)} characters")
        
        self.transcription = full_transcription
        return full_transcription
    
    def _recognize_with_gemini(self, audio_data):
        """使用Google Gemini API进行语音识别"""
        try:
            # 将音频数据保存为临时文件
            temp_file = os.path.join(self.config["temp_dir"], "temp_audio.wav")
            with open(temp_file, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # 使用FFmpeg将音频转换为base64
            import base64
            with open(temp_file, "rb") as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # 调用Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "请转录以下音频内容。如果是中文内容，请用中文回答；如果是英文内容，请用英文回答。只需要返回转录的文本，不需要任何其他内容。"
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": audio_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024
                }
            }
            
            # 构建URL，包含API密钥和模型名称（使用 Gemini 2.0）
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.config['api_key']}"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                response_json = response.json()
                if 'candidates' in response_json and len(response_json['candidates']) > 0:
                    content = response_json['candidates'][0]['content']['parts'][0]['text']
                    return content
                else:
                    logger.error(f"Gemini API响应格式不正确: {response_json}")
                    return ""
            else:
                logger.error(f"Gemini API请求失败: {response.status_code} {response.text}")
                return ""
        
        except Exception as e:
            logger.error(f"使用Gemini识别失败: {e}")
            return ""
    
    def _recognize_with_openai(self, audio_data):
        """使用OpenAI Whisper API进行语音识别"""
        if not self.has_api or self.config["api_type"] != "openai":
            return ""
        
        try:
            # 将音频数据保存为临时文件
            temp_file = os.path.join(self.config["temp_dir"], "temp_audio.wav")
            with open(temp_file, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # 使用OpenAI API
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}"
            }
            
            with open(temp_file, "rb") as f:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": f},
                    data={"model": "whisper-1"}
                )
            
            if response.status_code == 200:
                return response.json()["text"]
            else:
                logger.error(f"OpenAI API请求失败: {response.status_code} {response.text}")
                return ""
        
        except Exception as e:
            logger.error(f"使用OpenAI识别失败: {e}")
            return ""

    def extract_text_with_ocr(self):
        """使用OCR从视频帧中提取文本"""
        if not self.has_ocr:
            logger.warning("未安装pytesseract库，无法使用OCR文本识别功能。")
            return ""
        
        print("进行OCR文本识别... / Performing OCR text extraction...")
        
        # 提取关键帧
        if not self.key_frames:
            self.extract_key_frames()
        
        # 使用OCR识别文本
        ocr_results = []
        
        for i, frame in enumerate(self.key_frames):
            print(f"处理帧 {i+1}/{len(self.key_frames)}: {frame['time']:.1f}s")
            print(f"Processing frame {i+1}/{len(self.key_frames)}: {frame['time']:.1f}s")
            
            try:
                # 根据配置的语言选择OCR语言
                if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()):
                    lang = "chi_sim+eng"
                else:
                    lang = "eng"
                
                # 使用pytesseract进行OCR
                text = self.pytesseract.image_to_string(Image.open(frame["file"]), lang=lang)
                
                if text.strip():
                    ocr_results.append({
                        "time": frame["time"],
                        "text": text.strip()
                    })
            
            except Exception as e:
                logger.warning(f"OCR识别帧 {i+1} 失败: {e}")
                continue
        
        # 合并OCR结果
        ocr_text = "\n".join([f"[{result['time']:.1f}s] {result['text']}" for result in ocr_results])
        
        print(f"OCR识别完成，识别出 {len(ocr_results)} 个文本块")
        print(f"OCR recognition complete, recognized {len(ocr_results)} text blocks")
        
        self.ocr_text = ocr_text
        return ocr_text
    
    def _detect_chinese(self):
        """检测视频是否包含中文内容"""
        # 简单检测文件名是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', self.input_path):
            return True
        
        # 如果已经有OCR结果，检测OCR文本是否包含中文
        if hasattr(self, 'ocr_text') and self.ocr_text:
            if re.search(r'[\u4e00-\u9fff]', self.ocr_text):
                return True
        
        return False
    
    def generate_titles(self):
        """根据视频内容生成吸引眼球的标题"""
        print("生成标题... / Generating titles...")
        
        # 确保已经提取了文本内容
        if not hasattr(self, 'transcription') or not self.transcription:
            self.recognize_speech()
        
        if not hasattr(self, 'ocr_text') or not self.ocr_text:
            self.extract_text_with_ocr()
        
        # 合并所有文本
        all_text = f"{self.transcription}\n\n{self.ocr_text}"
        
        # 如果文本太少，无法生成有意义的标题
        if len(all_text.strip()) < 10:
            print("提取的文本内容太少，无法生成有意义的标题")
            print("Extracted text content is too limited to generate meaningful titles")
            
            # 使用文件名作为备选
            filename = os.path.basename(self.input_path)
            filename = os.path.splitext(filename)[0]
            filename = filename.replace("_", " ").replace("-", " ")
            
            titles = [
                f"精彩视频: {filename}",
                f"不容错过的精彩瞬间: {filename}",
                f"震撼视频集锦: {filename}"
            ]
            
            self.titles = titles
            return titles
        
        # 根据文本内容生成标题
        if self.has_api:
            if self.config["api_type"] == "gemini":
                titles = self._generate_titles_with_gemini(all_text)
            elif self.config["api_type"] == "openai":
                titles = self._generate_titles_with_openai(all_text)
            else:
                titles = self._generate_titles_with_rules(all_text)
        else:
            titles = self._generate_titles_with_rules(all_text)
        
        print(f"生成了 {len(titles)} 个标题")
        print(f"Generated {len(titles)} titles")
        
        for i, title in enumerate(titles):
            print(f"  {i+1}. {title}")
        
        self.titles = titles
        return titles
    
    def _generate_titles_with_openai(self, text):
        """使用OpenAI API生成标题"""
        try:
            # 准备提示词
            style = self.config["title_style"]
            language = "中文" if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()) else "英文"
            max_length = self.config["max_title_length"]
            count = self.config["title_count"]
            
            style_descriptions = {
                "exciting": "令人兴奋、充满吸引力的",
                "funny": "幽默搞笑的",
                "emotional": "情感丰富、打动人心的",
                "informative": "信息丰富、专业的"
            }
            
            style_desc = style_descriptions.get(style, "令人兴奋的")
            
            prompt = f"""
            根据以下视频内容，生成{count}个{style_desc}短视频标题，用于吸引观众点击。
            标题应该是{language}的，每个标题不超过{max_length}个字符。
            标题应该引人注目、吸引眼球，适合在短视频平台（如抖音、快手、TikTok）使用。
            不要使用数字编号，直接列出标题。
            
            视频内容：
            {text[:2000]}  # 限制文本长度以避免超出API限制
            """
            
            # 调用OpenAI API
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "你是一个专业的短视频标题生成助手，擅长创作吸引人的标题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # 处理返回的内容，提取标题
                titles = [line.strip() for line in content.split("\n") if line.strip()]
                # 过滤掉可能的编号
                titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
                return titles[:count]  # 确保返回指定数量的标题
            else:
                logger.error(f"OpenAI API请求失败: {response.status_code} {response.text}")
                # 回退到基于规则的方法
                return self._generate_titles_with_rules(text)
        
        except Exception as e:
            logger.error(f"使用OpenAI生成标题失败: {e}")
            # 回退到基于规则的方法
            return self._generate_titles_with_rules(text)
    
    def _generate_titles_with_gemini(self, text):
        """使用Google Gemini API生成标题"""
        try:
            print("正在使用Gemini API生成标题...")
            
            # 如果文本内容太少，使用文件名
            if len(text.strip()) < 50:
                filename = os.path.basename(self.input_path)
                filename = os.path.splitext(filename)[0]
                filename = filename.replace("_", " ").replace("-", " ")
                text = f"这是一个关于{filename}的视频"
            
            # 准备提示词
            style = self.config["title_style"]
            language = "中文" if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()) else "英文"
            max_length = self.config["max_title_length"]
            count = self.config["title_count"]
            
            style_descriptions = {
                "exciting": "令人兴奋、充满吸引力的",
                "funny": "幽默搞笑的",
                "emotional": "情感丰富、打动人心的",
                "informative": "信息丰富、专业的"
            }
            
            style_desc = style_descriptions.get(style, "令人兴奋的")
            
            prompt = f"""
            根据以下视频内容，生成{count}个{style_desc}短视频标题，用于吸引观众点击。
            标题应该是{language}的，每个标题不超过{max_length}个字符。
            标题应该引人注目、吸引眼球，适合在短视频平台（如抖音、快手、TikTok）使用。
            不要使用数字编号，直接列出标题。
            
            视频内容：
            {text[:1000]}
            """
            
            print(f"发送到Gemini API的提示词: {prompt[:100]}...")
            
            # 调用Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # 构建URL，包含API密钥和模型名称（使用 Gemini 2.0）
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.config['api_key']}"
            
            print(f"正在调用Gemini API (Gemini 2.0)...")
            response = requests.post(
                api_url,
                headers=headers,
                json=data
            )
            
            print(f"Gemini API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print(f"Gemini API响应: {json.dumps(response_json)[:200]}...")
                
                # 提取生成的文本
                if 'candidates' in response_json and len(response_json['candidates']) > 0:
                    content = response_json['candidates'][0]['content']['parts'][0]['text']
                    print(f"Gemini生成的内容: {content[:200]}...")
                    
                    # 处理返回的内容，提取标题
                    titles = [line.strip() for line in content.split("\n") if line.strip()]
                    # 过滤掉可能的编号
                    titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
                    
                    # 确保至少有count个标题
                    while len(titles) < count:
                        if language == "中文":
                            titles.append(f"精彩视频：{os.path.splitext(os.path.basename(self.input_path))[0]}")
                        else:
                            titles.append(f"Amazing video: {os.path.splitext(os.path.basename(self.input_path))[0]}")
                    
                    return titles[:count]  # 确保返回指定数量的标题
                else:
                    logger.error(f"Gemini API响应格式不正确: {response_json}")
                    print(f"Gemini API响应格式不正确，使用规则生成标题")
                    return self._generate_titles_with_rules(text)
            else:
                logger.error(f"Gemini API请求失败: {response.status_code} {response.text}")
                print(f"Gemini API请求失败: {response.status_code}，使用规则生成标题")
                # 回退到基于规则的方法
                return self._generate_titles_with_rules(text)
        
        except Exception as e:
            logger.error(f"使用Gemini生成标题失败: {e}")
            print(f"使用Gemini生成标题失败: {e}")
            import traceback
            traceback.print_exc()
            # 回退到基于规则的方法
            return self._generate_titles_with_rules(text)
    
    def _generate_titles_with_rules(self, text):
        """使用规则生成标题"""
        # 提取关键词
        words = text.split()
        # 过滤掉短词和常见词
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "like", "through", "over", "before", "after", "between", "under", "during", "without", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could"}
        keywords = [word for word in words if len(word) > 3 and word.lower() not in stopwords]
        
        # 如果关键词太少，使用文件名
        if len(keywords) < 3:
            filename = os.path.basename(self.input_path)
            filename = os.path.splitext(filename)[0]
            filename = filename.replace("_", " ").replace("-", " ")
            keywords.append(filename)
        
        # 选择最常见的几个关键词
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, _ in keyword_counts.most_common(5)]
        
        # 确保至少有一个关键词
        if not top_keywords:
            top_keywords = ["视频", "精彩", "瞬间"]
        
        # 根据语言选择模板
        if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()):
            templates = [
                "震撼！{keyword}的精彩瞬间",
                "不看后悔！{keyword}超燃集锦",
                "太精彩了！{keyword}让人惊叹",
                "这个{keyword}视频火了，网友：太厉害了",
                "一分钟看完{keyword}的精彩表现",
                "专业解析：为什么{keyword}如此与众不同",
                "独家揭秘：{keyword}背后的故事",
                "震惊全网：{keyword}的惊人表现",
                "这就是{keyword}的魅力所在！",
                "当{keyword}遇上{keyword2}，结果太震撼"
            ]
        else:
            templates = [
                "Incredible! Amazing moments of {keyword}",
                "Don't miss! Epic highlights of {keyword}",
                "Wow! {keyword} will leave you speechless",
                "This {keyword} video is going viral! Here's why",
                "One minute of pure {keyword} brilliance",
                "Professional analysis: Why {keyword} is so special",
                "Exclusive: The story behind {keyword}",
                "Internet sensation: {keyword}'s incredible performance",
                "This is why {keyword} is so captivating!",
                "When {keyword} meets {keyword2}, the result is amazing"
            ]
        
        # 生成标题
        import random
        titles = []
        count = min(self.config["title_count"], len(templates))
        
        for i in range(count):
            template = random.choice(templates)
            if templates:  # 确保templates不为空
                templates.remove(template)  # 避免重复
            
            # 替换关键词
            if top_keywords:
                keyword = random.choice(top_keywords)
                title = template.replace("{keyword}", keyword)
                
                # 如果模板包含第二个关键词占位符
                if "{keyword2}" in title and len(top_keywords) > 1:
                    remaining_keywords = [k for k in top_keywords if k != keyword]
                    if remaining_keywords:  # 确保remaining_keywords不为空
                        keyword2 = random.choice(remaining_keywords)
                        title = title.replace("{keyword2}", keyword2)
                    else:
                        title = title.replace("{keyword2}", "精彩内容")
                elif "{keyword2}" in title:
                    title = title.replace("{keyword2}", "精彩内容")
            else:
                # 如果没有关键词，使用通用标题
                title = template.replace("{keyword}", "精彩视频").replace("{keyword2}", "精彩内容")
            
            titles.append(title)
        
        # 如果没有生成足够的标题，添加一些通用标题
        while len(titles) < self.config["title_count"]:
            if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()):
                titles.append(f"精彩视频：{os.path.splitext(os.path.basename(self.input_path))[0]}")
            else:
                titles.append(f"Amazing video: {os.path.splitext(os.path.basename(self.input_path))[0]}")
        
        return titles
    
    def generate_thumbnails(self):
        """生成视频封面"""
        print("生成封面... / Generating thumbnails...")
        
        # 确保已经提取了关键帧
        if not self.key_frames:
            self.extract_key_frames()
        
        # 创建封面目录
        thumbnails_dir = os.path.join(self.output_dir, "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)
        
        thumbnails = []
        count = min(self.config["thumbnail_count"], len(self.key_frames))
        
        # 选择最有代表性的帧作为封面
        selected_frames = self._select_best_frames(count)
        
        # 如果有API密钥且类型为gemini，使用Gemini API生成封面标题
        if self.has_api and self.config["api_type"] == "gemini" and hasattr(self, 'transcription') and self.transcription:
            print("使用Gemini API生成封面标题...")
            cover_titles = self._generate_cover_titles_with_gemini(self.transcription)
        else:
            cover_titles = self.titles
        
        for i, frame in enumerate(selected_frames):
            print(f"处理封面 {i+1}/{count}")
            print(f"Processing thumbnail {i+1}/{count}")
            
            # 生成封面
            thumbnail_file = os.path.join(thumbnails_dir, f"thumbnail_{i+1}.jpg")
            
            # 根据配置的风格处理封面
            title_to_use = cover_titles[i] if cover_titles and i < len(cover_titles) else None
            self._process_thumbnail(frame["file"], thumbnail_file, i, title_to_use)
            
            thumbnails.append({
                "file": thumbnail_file,
                "time": frame["time"]
            })
        
        print(f"生成了 {len(thumbnails)} 个封面")
        print(f"Generated {len(thumbnails)} thumbnails")
        
        self.thumbnails = thumbnails
        return thumbnails
    
    def _generate_cover_titles_with_gemini(self, text):
        """使用Google Gemini API生成封面标题"""
        try:
            print("正在使用Gemini API生成封面标题...")
            
            # 如果文本内容太少，使用文件名
            if len(text.strip()) < 50:
                filename = os.path.basename(self.input_path)
                filename = os.path.splitext(filename)[0]
                filename = filename.replace("_", " ").replace("-", " ")
                text = f"这是一个关于{filename}的视频"
            
            # 准备提示词
            count = self.config["thumbnail_count"]
            language = "中文" if self.config["language"] == "zh" or (self.config["language"] == "auto" and self._detect_chinese()) else "英文"
            
            prompt = f"""
            根据以下视频内容，生成{count}个简短但有吸引力的封面标题。
            这些标题将直接显示在视频封面上，所以应该简短、有力、引人注目。
            每个标题不超过20个字符，应该是{language}的。
            不要使用数字编号，直接列出标题。
            
            视频内容：
            {text[:1000]}
            """
            
            print(f"发送到Gemini API的提示词: {prompt[:100]}...")
            
            # 调用Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # 构建URL，包含API密钥和模型名称（使用 Gemini 2.0）
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.config['api_key']}"
            
            print(f"正在调用Gemini API (Gemini 2.0)...")
            response = requests.post(
                api_url,
                headers=headers,
                json=data
            )
            
            print(f"Gemini API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print(f"Gemini API响应: {json.dumps(response_json)[:200]}...")
                
                # 提取生成的文本
                if 'candidates' in response_json and len(response_json['candidates']) > 0:
                    content = response_json['candidates'][0]['content']['parts'][0]['text']
                    print(f"Gemini生成的封面标题: {content[:200]}...")
                    
                    # 处理返回的内容，提取标题
                    titles = [line.strip() for line in content.split("\n") if line.strip()]
                    # 过滤掉可能的编号
                    titles = [re.sub(r'^\d+\.\s*', '', title) for title in titles]
                    
                    # 确保至少有count个标题
                    while len(titles) < count:
                        if language == "中文":
                            titles.append("精彩瞬间")
                        else:
                            titles.append("Amazing Moment")
                    
                    return titles[:count]  # 确保返回指定数量的标题
                else:
                    logger.error(f"Gemini API响应格式不正确: {response_json}")
                    print(f"Gemini API响应格式不正确，使用普通标题")
                    return self.titles if hasattr(self, 'titles') and self.titles else None
            else:
                logger.error(f"Gemini API请求失败: {response.status_code} {response.text}")
                print(f"Gemini API请求失败: {response.status_code}，使用普通标题")
                return self.titles if hasattr(self, 'titles') and self.titles else None
        
        except Exception as e:
            logger.error(f"使用Gemini生成封面标题失败: {e}")
            print(f"使用Gemini生成封面标题失败: {e}")
            import traceback
            traceback.print_exc()
            return self.titles if hasattr(self, 'titles') and self.titles else None
    
    def _select_best_frames(self, count):
        """选择最有代表性的帧作为封面"""
        # 如果关键帧数量不足，返回所有关键帧
        if len(self.key_frames) <= count:
            return self.key_frames
        
        # 计算每个帧的分数
        frame_scores = []
        
        for frame in self.key_frames:
            score = 0
            
            # 读取图像
            img = cv2.imread(frame["file"])
            if img is None:
                continue
            
            # 1. 亮度和对比度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 适中的亮度和高对比度得分高
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(contrast / 50, 1.0)
            
            # 2. 颜色丰富度
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            saturation_score = min(saturation / 128, 1.0)
            
            # 3. 人脸检测
            face_score = 0
            if self.has_cv2:
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        face_score = min(len(faces) * 0.5, 1.0)
                except Exception as e:
                    logger.warning(f"人脸检测失败: {e}")
            
            # 4. 边缘检测（复杂度）
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / (img.shape[0] * img.shape[1])
            edge_score = min(edge_density * 5, 1.0)
            
            # 综合评分
            score = brightness_score * 0.2 + contrast_score * 0.3 + saturation_score * 0.2 + face_score * 0.2 + edge_score * 0.1
            
            frame_scores.append((frame, score))
        
        # 按分数排序并选择最高分的帧
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected_frames = [item[0] for item in frame_scores[:count]]
        
        return selected_frames
    
    def _process_thumbnail(self, input_file, output_file, index, custom_title=None):
        """处理封面图像"""
        try:
            # 打开图像
            img = Image.open(input_file)
            
            # 调整大小（保持宽高比）
            max_size = (1280, 720)
            img.thumbnail(max_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # 根据配置的风格处理封面
            style = self.config["cover_style"]
            
            if style == "modern":
                # 增强对比度和饱和度
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.3)
                
                # 添加文字
                title = custom_title if custom_title else (self.titles[index] if self.titles and len(self.titles) > index else None)
                if title:
                    img = self._add_text_to_image(img, title)
            
            elif style == "minimal":
                # 转换为黑白
                img = img.convert("L")
                img = Image.merge("RGB", [img, img, img])
                
                # 增强对比度
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.4)
                
                # 添加文字
                title = custom_title if custom_title else (self.titles[index] if self.titles and len(self.titles) > index else None)
                if title:
                    img = self._add_text_to_image(img, title, color=(255, 255, 255))
            
            elif style == "dramatic":
                # 增强对比度和锐度
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.5)
                
                # 添加暗角效果
                img = self._add_vignette(img)
                
                # 添加文字
                title = custom_title if custom_title else (self.titles[index] if self.titles and len(self.titles) > index else None)
                if title:
                    img = self._add_text_to_image(img, title, color=(255, 255, 255))
            
            elif style == "colorful":
                # 增强饱和度
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.6)
                
                # 添加文字
                title = custom_title if custom_title else (self.titles[index] if self.titles and len(self.titles) > index else None)
                if title:
                    img = self._add_text_to_image(img, title, color=(255, 255, 0))
            
            # 保存处理后的图像
            img.save(output_file, quality=95)
        
        except Exception as e:
            logger.error(f"处理封面失败: {e}")
            # 如果处理失败，复制原始图像
            shutil.copy(input_file, output_file)
    
    def _add_text_to_image(self, img, text, color=(255, 255, 255)):
        """向图像添加文字"""
        # 创建可绘制对象
        draw = ImageDraw.Draw(img)
        
        # 设置字体
        font_size = int(img.width * 0.05)  # 字体大小为图像宽度的5%
        
        try:
            if self.config["font_path"]:
                font = ImageFont.truetype(self.config["font_path"], font_size)
            else:
                # 尝试使用系统字体
                try:
                    if os.name == 'nt':  # Windows
                        font = ImageFont.truetype("arial.ttf", font_size)
                    else:  # Linux/Mac
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 文本换行
        max_width = int(img.width * 0.9)  # 最大宽度为图像宽度的90%
        lines = textwrap.wrap(text, width=30)  # 大约30个字符一行
        
        # 计算文本总高度
        line_height = font_size * 1.2
        text_height = len(lines) * line_height
        
        # 计算文本位置（底部居中）
        y = img.height - text_height - int(img.height * 0.05)  # 距离底部5%
        
        # 添加半透明背景
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            [(0, y - line_height * 0.5), (img.width, y + text_height + line_height * 0.5)],
            fill=(0, 0, 0, 128)
        )
        
        # 将背景合并到原图
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')  # 转回RGB模式
        
        # 重新创建可绘制对象
        draw = ImageDraw.Draw(img)
        
        # 绘制文本
        for i, line in enumerate(lines):
            line_y = y + i * line_height
            
            # 使用 getbbox 或 getsize 替代 textsize（根据 Pillow 版本）
            try:
                # 新版 Pillow
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
            except AttributeError:
                try:
                    # 旧版 Pillow
                    line_width, _ = font.getsize(line)
                except:
                    # 最后的备选方案
                    line_width = len(line) * font_size * 0.6
            
            x = (img.width - line_width) / 2  # 居中
            
            # 绘制文本阴影
            draw.text((x+2, line_y+2), line, font=font, fill=(0, 0, 0))
            # 绘制文本
            draw.text((x, line_y), line, font=font, fill=color)
        
        return img
    
    def _add_vignette(self, img):
        """添加暗角效果"""
        # 创建椭圆形蒙版
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 绘制从中心向外渐变的椭圆
        for i in range(min(img.size) // 2, 0, -1):
            # 计算椭圆的边界框
            x0 = (img.width - i * 2) // 2
            y0 = (img.height - i * 2) // 2
            x1 = x0 + i * 2
            y1 = y0 + i * 2
            
            # 计算透明度（从中心向外逐渐变暗）
            alpha = int(255 * (i / (min(img.size) // 2)) ** 0.5)
            
            # 绘制椭圆
            draw.ellipse((x0, y0, x1, y1), fill=alpha)
        
        # 应用蒙版
        img = img.convert('RGBA')
        mask = mask.resize(img.size)
        
        # 创建暗色层
        black = Image.new('RGBA', img.size, (0, 0, 0, 255))
        
        # 使用蒙版混合原图和暗色层
        result = Image.composite(img, black, mask)
        
        return result.convert('RGB')
    
    def analyze_and_generate(self):
        """分析视频并生成内容"""
        try:
            print("开始分析视频并生成内容... / Starting video analysis and content generation...")
            
            # 获取视频信息
            self.get_video_info()
            
            # 提取关键帧
            self.extract_key_frames()
            
            # 语音识别 - 确保不跳过
            if self.config["speech_recognition"]:
                print("正在进行语音识别，这可能需要一些时间...")
                self.recognize_speech()  # 修改后的方法会处理没有安装库的情况
            else:
                print("跳过语音识别")
                # 即使跳过语音识别，也使用文件名作为文本内容
                filename = os.path.basename(self.input_path)
                filename = os.path.splitext(filename)[0]
                filename = filename.replace("_", " ").replace("-", " ")
                self.transcription = f"这是一个关于{filename}的视频"
            
            # OCR文本识别
            if self.config["ocr"] and self.has_ocr:
                print("正在进行OCR文本识别，这可能需要一些时间...")
                self.extract_text_with_ocr()
            else:
                print("跳过OCR文本识别")
                self.ocr_text = ""
            
            # 生成标题
            self.generate_titles()
            
            # 生成封面
            self.generate_thumbnails()
            
            # 保存结果
            self._save_results()
            
            print("内容生成完成！/ Content generation complete!")
            print(f"标题和封面已保存到: {self.output_dir}")
            print(f"Titles and thumbnails saved to: {self.output_dir}")
            
            return {
                "titles": self.titles,
                "thumbnails": [t["file"] for t in self.thumbnails]
            }
        
        except Exception as e:
            logger.error(f"分析和生成内容时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self):
        """保存生成结果"""
        # 保存标题
        titles_file = os.path.join(self.output_dir, "titles.txt")
        with open(titles_file, "w", encoding="utf-8") as f:
            for i, title in enumerate(self.titles):
                f.write(f"{i+1}. {title}\n")
        
        # 保存文本内容
        if hasattr(self, 'transcription') and self.transcription:
            transcript_file = os.path.join(self.output_dir, "transcription.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(self.transcription)
        
        if hasattr(self, 'ocr_text') and self.ocr_text:
            ocr_file = os.path.join(self.output_dir, "ocr_text.txt")
            with open(ocr_file, "w", encoding="utf-8") as f:
                f.write(self.ocr_text)
        
        # 保存元数据
        metadata = {
            "video_file": self.input_path,
            "duration": self.duration,
            "resolution": f"{self.width}x{self.height}",
            "titles": self.titles,
            "thumbnails": [t["file"] for t in self.thumbnails]
        }
        
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="内容生成器 - 自动分析视频内容并生成吸引眼球的标题和封面")
    
    # 必需参数
    parser.add_argument("input", help="输入视频文件路径")
    
    # 可选参数
    parser.add_argument("-o", "--output", help="输出目录路径")
    parser.add_argument("--frame-interval", type=int, default=5, help="截取帧的间隔（秒），默认为5")
    parser.add_argument("--no-speech", action="store_false", dest="speech_recognition", help="禁用语音识别")
    parser.add_argument("--no-ocr", action="store_false", dest="ocr", help="禁用OCR文本识别")
    parser.add_argument("--title-style", choices=["exciting", "funny", "emotional", "informative"], default="exciting", help="标题风格")
    parser.add_argument("--cover-style", choices=["modern", "minimal", "dramatic", "colorful"], default="modern", help="封面风格")
    parser.add_argument("--language", choices=["auto", "zh", "en"], default="auto", help="语言")
    parser.add_argument("--max-title-length", type=int, default=50, help="标题最大长度")
    parser.add_argument("--api-key", help="API密钥")
    parser.add_argument("--font-path", help="字体路径")
    parser.add_argument("--use-gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--thumbnail-count", type=int, default=5, help="生成的候选封面数量")
    parser.add_argument("--title-count", type=int, default=3, help="生成的候选标题数量")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    # 处理输出目录
    output_dir = args.output
    if output_dir:
        # 确保输出目录是绝对路径
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
        
        # 尝试创建输出目录
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"使用指定的输出目录: {output_dir}")
        except Exception as e:
            print(f"创建指定的输出目录时出错: {e}")
            # 使用输入文件所在目录作为备选
            output_dir = os.path.dirname(os.path.abspath(args.input))
            print(f"使用输入文件所在目录作为备选: {output_dir}")
    else:
        # 使用输入文件所在目录
        output_dir = os.path.dirname(os.path.abspath(args.input))
        print(f"使用输入文件所在目录: {output_dir}")
    
    # 检查依赖库
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        missing_deps.append("pillow")
    
    if args.speech_recognition:
        try:
            import speech_recognition
        except ImportError:
            missing_deps.append("SpeechRecognition")
    
    if args.ocr:
        try:
            import pytesseract
        except ImportError:
            missing_deps.append("pytesseract")
    
    # 检查FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        missing_deps.append("ffmpeg (需要安装到系统PATH)")
    
    if missing_deps:
        print("缺少以下依赖库，请安装后再运行:")
        print("Missing dependencies, please install them before running:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        if "opencv-python" in missing_deps or "pillow" in missing_deps:
            print("\n这些是必需的依赖库，无法继续执行。")
            print("These are required dependencies, cannot continue.")
            return 1
        else:
            print("\n将在有限功能模式下继续执行。")
            print("Will continue in limited functionality mode.")
    
    # 创建ContentGenerator实例
    try:
        generator = ContentGenerator(
            input_path=args.input,
            output_dir=output_dir,
            frame_interval=args.frame_interval,
            speech_recognition=args.speech_recognition,
            ocr=args.ocr,
            title_style=args.title_style,
            cover_style=args.cover_style,
            language=args.language,
            max_title_length=args.max_title_length,
            api_key=args.api_key,
            font_path=args.font_path,
            use_gpu=args.use_gpu,
            thumbnail_count=args.thumbnail_count,
            title_count=args.title_count
        )
        
        # 分析视频并生成内容
        generator.analyze_and_generate()
        return 0
    except Exception as e:
        print(f"错误: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
