# cv2_patch.py - Patch for handling OpenCV circular imports
import sys
import types

def patch_cv2_modules():
    """
    Create mock modules to prevent circular import errors in OpenCV.
    This function creates dummy modules and classes for the problematic parts of OpenCV.
    """
    # Create mock modules
    for module_name in [
        'cv2.gapi', 
        'cv2.typing', 
        'cv2.dnn', 
        'cv2.gapi.wip', 
        'cv2.gapi.wip.draw', 
        'cv2.gapi_wip_gst_GStreamerPipeline'
    ]:
        if module_name not in sys.modules:
            dummy_module = types.ModuleType(module_name)
            sys.modules[module_name] = dummy_module
    
    # Add missing attributes to the gapi.wip.draw module
    draw_module = sys.modules.get('cv2.gapi.wip.draw')
    if draw_module:
        class DummyClass:
            def __getattr__(self, name):
                return DummyClass()
            
            def __call__(self, *args, **kwargs):
                return DummyClass()
        
        # Add all the missing attributes mentioned in the error
        draw_module.Text = DummyClass()
        draw_module.Circle = DummyClass()
        draw_module.Image = DummyClass()
        draw_module.Line = DummyClass()
        draw_module.Rect = DummyClass()
        draw_module.Mosaic = DummyClass()
        draw_module.Poly = DummyClass()
    
    # Handle gapi module
    gapi_module = sys.modules.get('cv2.gapi')
    if gapi_module:
        class DummyClass:
            def __getattr__(self, name):
                return DummyClass()
            
            def __call__(self, *args, **kwargs):
                return DummyClass()
        
        # Add missing attributes
        gapi_module.wip = DummyClass()
        setattr(gapi_module.wip, 'draw', DummyClass())
        setattr(gapi_module.wip, 'GStreamerPipeline', DummyClass())
        print("OpenCV modules patched to prevent circular imports")