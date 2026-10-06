"""桥接脚本共用的轻量 Windows 运行时初始化工具。"""

import os
import sys


def configure_windows_dll_path():
    """注册 Python DLL 目录并移除已知冲突的 PATH 项。"""
    if sys.platform != "win32":
        return
    try:
        py_dir = os.path.dirname(sys.executable)
        dll_dir = os.path.join(py_dir, "DLLs")
        for directory in (py_dir, dll_dir):
            if os.path.isdir(directory):
                try:
                    os.add_dll_directory(directory)
                except Exception:
                    pass
        raw_path = os.environ.get("PATH", "").split(os.pathsep)
        clean_path = [
            entry for entry in raw_path
            if not any(token in entry.lower() for token in (
                "anaconda", "miniconda", "cuda\\v11", "cuda/v11", "openvino"
            ))
        ]
        for directory in (py_dir, dll_dir):
            if os.path.isdir(directory) and directory not in clean_path:
                clean_path.insert(0, directory)
        for variable in list(os.environ):
            if "CUDA" in variable and "v11" in os.environ[variable].lower():
                del os.environ[variable]
        os.environ["PATH"] = os.pathsep.join(clean_path)
    except Exception:
        pass
