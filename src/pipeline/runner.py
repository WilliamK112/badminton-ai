"""
主流程运行器 - 一键运行完整分析流水线
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.pipeline import IntegratedPipeline, RallyAnalyzer
from src.utils.video import VideoReader, VideoWriter


def run_pipeline(
    video_path: str,
    output_dir: str = "output",
    sample_interval: int = 10,
    max_frames: int = 5000,
    visualize: bool = True
):
    """运行完整流水线"""
    
    print("=" * 60)
    print("Badminton AI - 整合分析流水线")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. 初始化流水线
    print("\n[1/4] 初始化模块...")
    pipeline = IntegratedPipeline(
        player_model="yolo11n.pt",
        shuttle_model="yolo11n.pt", 
        pose_model="yolo11n-pose.pt"
    )
    print("  ✓ 球员追踪器")
    print("  ✓ 羽毛球追踪器")
    print("  ✓ 骨架检测器")
    
    # 2. 处理视频
    print(f"\n[2/4] 处理视频: {video_path}")
    frame_data = pipeline.process_video(
        video_path,
        sample_interval=sample_interval,
        output_path=str(output_path / "analysis.json"),
        max_frames=max_frames
    )
    print(f"  ✓ 处理了 {len(frame_data)} 帧")
    
    # 3. 分析回合
    print("\n[3/4] 分析回合...")
    features = RallyAnalyzer.extract_features(frame_data)
    print(f"  球员检测率: {features.get('player_detection_rate', 0):.2%}")
    print(f"  羽毛球检测率: {features.get('shuttle_detection_rate', 0):.2%}")
    
    # 4. 生成可视化
    if visualize:
        print("\n[4/4] 生成可视化...")
        _generate_visualization(pipeline, video_path, frame_data, output_path)
    
    print("\n" + "=" * 60)
    print("✅ 完成! 输出目录:", output_path)
    print("=" * 60)
    
    return pipeline, frame_data


def _generate_visualization(pipeline, video_path: str, frame_data, output_path: Path):
    """生成可视化视频"""
    
    with VideoReader(video_path) as reader:
        output_video = str(output_path / "output.mp4")
        
        with VideoWriter(
            output_video,
            fps=reader.fps,
            frame_size=(reader.width, reader.height)
        ) as writer:
            import numpy as np
            
            for i, fd in enumerate(frame_data[:100]):  # 只可视化前100帧
                frame = reader.read_frame(fd.frame_idx)
                if frame is None:
                    continue
                    
                frame = pipeline.visualize_frame(frame, fd)
                writer.write(frame)
                
                if i % 20 == 0:
                    print(f"  已可视化 {i} 帧")
    
    print(f"  ✓ 可视化已保存: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Badminton AI 流水线")
    parser.add_argument("video", nargs="?", default="thisone.mp4", help="输入视频路径")
    parser.add_argument("--output", "-o", default="output", help="输出目录")
    parser.add_argument("--interval", "-i", type=int, default=10, help="采样间隔")
    parser.add_argument("--max-frames", "-m", type=int, default=5000, help="最大处理帧数")
    parser.add_argument("--no-visualize", action="store_true", help="不生成可视化")
    
    args = parser.parse_args()
    
    run_pipeline(
        args.video,
        args.output,
        args.interval,
        args.max_frames,
        not args.no_visualize
    )


if __name__ == "__main__":
    main()