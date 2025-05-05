#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoCut示例脚本 - 演示如何使用AutoCut工具
"""

import os
import sys
from autocut import AutoCut

def main():
    """示例脚本入口函数"""
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python example.py <演唱会视频文件路径>")
        return 1
    
    input_video = sys.argv[1]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_video):
        print(f"错误: 输入文件不存在: {input_video}")
        return 1
    
    # 设置输出路径
    output_dir = os.path.dirname(os.path.abspath(input_video))
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = os.path.join(output_dir, f"{base_name}_highlights.mp4")
    
    print(f"开始处理视频: {input_video}")
    print(f"输出文件将保存到: {output_video}")
    
    # 创建自定义配置
    config = {
        "min_clip_duration": 3,      # 最小片段时长3秒
        "max_clip_duration": 20,     # 最大片段时长20秒
        "energy_threshold": 0.65,    # 能量阈值0.65
        "applause_detection": True,  # 启用掌声检测
        "tempo_change_detection": True,  # 启用节奏变化检测
        "highlight_count": 8,        # 提取8个高潮片段
        "fade_duration": 0.8,        # 淡入淡出时长0.8秒
    }
    
    try:
        # 创建AutoCut实例并处理视频
        autocut = AutoCut(input_video, output_video, config)
        result_path = autocut.process()
        
        print("\n处理完成!")
        print(f"精彩集锦视频已保存到: {result_path}")
        print(f"分析图表已保存到: {os.path.splitext(result_path)[0]}_analysis.png")
        
        return 0
    except Exception as e:
        print(f"处理失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
