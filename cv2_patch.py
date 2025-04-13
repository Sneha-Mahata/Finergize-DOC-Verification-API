# cv2_patch.py
import sys
import types

# Create mock modules to prevent import errors
for module_name in ['cv2.gapi', 'cv2.typing', 'cv2.dnn', 'cv2.gapi.wip', 'cv2.gapi.wip.draw']:
    if module_name not in sys.modules:
        dummy_module = types.ModuleType(module_name)
        sys.modules[module_name] = dummy_module

# Add missing attributes to the gapi.wip.draw module
draw_module = sys.modules.get('cv2.gapi.wip.draw')
if draw_module:
    # Add all the missing attributes mentioned in the error
    class DummyClass:
        pass
    
    draw_module.Text = DummyClass
    draw_module.Circle = DummyClass
    draw_module.Image = DummyClass
    draw_module.Line = DummyClass
    draw_module.Rect = DummyClass
    draw_module.Mosaic = DummyClass
    draw_module.Poly = DummyClass