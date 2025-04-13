# cv2_patch.py
import sys
import types

# Create dummy modules to prevent import errors
for module_name in ['cv2.gapi', 'cv2.typing', 'cv2.dnn']:
    dummy_module = types.ModuleType(module_name)
    sys.modules[module_name] = dummy_module

# Enhance the dummy DNN module
if 'cv2.dnn' in sys.modules:
    dnn_module = sys.modules['cv2.dnn']
    dnn_module.DictValue = object  # Add missing DictValue class