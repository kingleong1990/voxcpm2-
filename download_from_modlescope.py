import os
import sys
from pathlib import Path
import subprocess


def check_modlescope_availability():
    """
    检查ModelScope是否可用
    """
    try:
        import modelscope
        print("ModelScope已安装")
        return True
    except ImportError:
        print("ModelScope未安装，需要先安装...")
        return False


def list_available_models():
    """
    不再列出ModelScope模型，因为我们要使用HF Hub上的VoxCPM2
    """
    print("此脚本现在将从ModelScope下载OpenBMB/VoxCPM2模型...")
    return []


def download_voxcpm_from_modlescope():
    """
    使用modelscope命令下载OpenBMB/VoxCPM2模型
    """
    print("正在从ModelScope下载OpenBMB/VoxCPM2模型...")
    
    # 创建模型存储目录 - 下载到当前目录下的models文件夹
    cache_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 构建下载命令
        cmd = [
            sys.executable, "-m", "modelscope", "download", 
            "--model", "OpenBMB/VoxCPM2",
            "--local-dir", os.path.join(cache_dir, "VoxCPM2")
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("模型下载成功!")
        model_dir = os.path.join(cache_dir, "VoxCPM2")
        print(f"模型已保存到: {model_dir}")
        return model_dir
        
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None
    except Exception as e:
        print(f"下载过程中发生未知错误: {str(e)}")
        return None


def main():
    print("检查ModelScope可用性...")
    if not check_modlescope_availability():
        print("请先安装ModelScope: python -m pip install modelscope")
        return False
    
    print("\n准备下载OpenBMB/VoxCPM2模型...")
    list_available_models()
    
    print("\n尝试下载VoxCPM2模型...")
    model_path = download_voxcpm_from_modlescope()
    
    if model_path:
        print(f"\n✅ 模型下载完成！")
        print(f"模型路径: {model_path}")
        print(f"运行Web应用: python app.py --model-dir \"{model_path}\"")
        return True
    else:
        print("\n❌ 无法从ModelScope下载VoxCPM2模型")
        print("\n💡 建议尝试以下方法:")
        print("   1. 检查网络连接")
        print("   2. 确认modelscope命令行工具已正确安装")
        print("   3. 手动执行命令: modelscope download --model OpenBMB/VoxCPM2")
        print("   4. 联系网络管理员检查是否有访问限制")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n所有下载尝试均已失败")
        sys.exit(1)