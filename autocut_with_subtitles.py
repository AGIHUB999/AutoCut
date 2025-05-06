#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoCut With Subtitles - 自动剪辑演唱会高潮部分并添加字幕的集成工具
"""

import os
import sys
import argparse
import logging
from typing import Dict
from pathlib import Path

from autocut import AutoCut
from subtitle_generator import SubtitleGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoCutWithSubtitles")

def process_video_with_subtitles(input_path: str, output_path: str, 
                                autocut_config: Dict = None, 
                                subtitle_config: Dict = None):
    """
    处理视频：先剪辑高潮部分，然后添加字幕
    
    Args:
        input_path: 输入视频文件路径
        output_path: 最终输出视频文件路径
        autocut_config: AutoCut配置参数
        subtitle_config: SubtitleGenerator配置参数
        
    Returns:
        str: 最终输出视频路径
    """
    try:
        # 步骤1: 使用AutoCut剪辑高潮部分
        logger.info("第1步: 剪辑视频高潮部分...")
        
        # 为AutoCut创建临时输出路径
        temp_output_dir = os.path.dirname(os.path.abspath(output_path))
        temp_output_filename = f"temp_highlight_{os.path.basename(output_path)}"
        temp_output_path = os.path.join(temp_output_dir, temp_output_filename)
        
        # 创建并运行AutoCut
        autocut = AutoCut(input_path, temp_output_path, autocut_config)
        highlight_path = autocut.process()
        
        logger.info(f"高潮剪辑完成: {highlight_path}")
        
        # 步骤2: 为剪辑后的视频添加字幕
        logger.info("第2步: 为高潮视频添加字幕...")
        
        # 创建并运行SubtitleGenerator
        subtitle_generator = SubtitleGenerator(highlight_path, output_path, subtitle_config)
        final_output_path = subtitle_generator.generate_subtitles()
        
        logger.info(f"字幕添加完成: {final_output_path}")
        
        # 清理临时文件
        if os.path.exists(highlight_path) and os.path.exists(final_output_path):
            os.remove(highlight_path)
            logger.info(f"已删除临时文件: {highlight_path}")
        
        return final_output_path
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="AutoCut With Subtitles - 自动剪辑演唱会高潮部分并添加字幕")
    
    # 基本参数
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", help="输出视频文件路径")
    
    # AutoCut参数
    parser.add_argument("--min-duration", type=float, default=5, help="最小片段时长(秒)")
    parser.add_argument("--max-duration", type=float, default=30, help="最大片段时长(秒)")
    parser.add_argument("--threshold", type=float, default=0.7, help="能量阈值(0-1)")
    parser.add_argument("--count", type=int, default=5, help="要提取的高潮片段数量")
    parser.add_argument("--no-applause", action="store_false", dest="applause", help="禁用掌声检测")
    parser.add_argument("--no-tempo", action="store_false", dest="tempo", help="禁用节奏变化检测")
    
    # 字幕参数
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
    
    # 创建AutoCut配置
    autocut_config = {
        "min_clip_duration": args.min_duration,
        "max_clip_duration": args.max_duration,
        "energy_threshold": args.threshold,
        "applause_detection": args.applause,
        "tempo_change_detection": args.tempo,
        "highlight_count": args.count,
    }
    
    # 创建SubtitleGenerator配置
    subtitle_config = {
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
    
    # 处理视频
    try:
        output_path = process_video_with_subtitles(
            args.input, 
            args.output, 
            autocut_config, 
            subtitle_config
        )
        print(f"处理完成! 最终输出文件: {output_path}")
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
