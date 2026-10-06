import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "App_NovaVision" / "ViTooAI_Bridge"))
from bridge_common import configure_windows_dll_path


class BridgeCommonTests(unittest.TestCase):
    def test_windows_path_cleanup_and_python_dll_registration(self):
        python_dir = r"C:\portable python"
        dll_dir = os.path.join(python_dir, "DLLs")
        original_path = os.pathsep.join((r"C:\safe", r"C:\Anaconda\bin", r"C:\CUDA\v11\bin", r"C:\OpenVINO\bin"))
        env = {"PATH": original_path, "CUDA_PATH_V11": r"C:\CUDA\v11"}
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(sys, "executable", os.path.join(python_dir, "python.exe")), \
             mock.patch("bridge_common.os.path.isdir", return_value=True), \
             mock.patch("bridge_common.os.add_dll_directory") as add_dll_directory, \
             mock.patch.dict(os.environ, env, clear=True):
            configure_windows_dll_path()
            add_dll_directory.assert_has_calls([mock.call(python_dir), mock.call(dll_dir)])
            path_parts = os.environ["PATH"].split(os.pathsep)
            self.assertEqual(path_parts[:2], [dll_dir, python_dir])
            self.assertIn(r"C:\safe", path_parts)
            self.assertNotIn("CUDA_PATH_V11", os.environ)
            self.assertFalse(any("anaconda" in part.lower() or "openvino" in part.lower() for part in path_parts))


if __name__ == "__main__":
    unittest.main()
