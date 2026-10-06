import os
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / "App_NovaVision" / "ViTooAI_Bridge"


class BridgeEntrypointTests(unittest.TestCase):
    def test_root_entries_preserve_execution_name_and_module_namespace(self):
        for module_name in ("__main__", "__mp_main__", "bridge_train"):
            for wrapper in sorted((ROOT / "ViTooAI_Bridge").glob("bridge_*.py")):
                with self.subTest(wrapper=wrapper.name, module_name=module_name):
                    source = wrapper.read_text(encoding="utf-8")
                    namespace = {"__file__": str(wrapper), "__name__": module_name}
                    original_cwd = os.getcwd()
                    original_argv = list(sys.argv)
                    target_namespace = {"__name__": module_name, "__file__": str(BRIDGE / wrapper.name), "main": object()}
                    with mock.patch.object(runpy, "run_path", return_value=target_namespace) as run_target:
                        exec(compile(source, str(wrapper), "exec"), namespace)
                    target = BRIDGE / wrapper.name
                    run_target.assert_called_once_with(str(target), run_name=module_name)
                    self.assertIn(str(BRIDGE), sys.path)
                    self.assertTrue(target.is_file())
                    self.assertEqual(os.getcwd(), original_cwd)
                    self.assertEqual(sys.argv, original_argv)
                    self.assertEqual(namespace["__file__"], str(wrapper))
                    self.assertEqual(namespace["__name__"], module_name)
                    self.assertIn("main", namespace)


if __name__ == "__main__":
    unittest.main()
