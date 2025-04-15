import sys
import types
import importlib

def patch_ultralytics_modules():
    """Creates missing module structure for ultralytics to load older models"""
    import ultralytics
    
    # Create main module structure if it doesn't exist
    if not hasattr(ultralytics, 'nn'):
        ultralytics.nn = types.ModuleType('ultralytics.nn')
        sys.modules['ultralytics.nn'] = ultralytics.nn
    
    if not hasattr(ultralytics.nn, 'modules'):
        ultralytics.nn.modules = types.ModuleType('ultralytics.nn.modules')
        sys.modules['ultralytics.nn.modules'] = ultralytics.nn.modules
    
    # Add missing modules
    module_names = [
        'conv',
        'block',
        'head',
        'transformer',
        'attention',
        'bottle',
        'upsample',
        'activation'
    ]
    
    for module_name in module_names:
        full_module_name = f'ultralytics.nn.modules.{module_name}'
        if full_module_name not in sys.modules:
            new_module = types.ModuleType(full_module_name)
            sys.modules[full_module_name] = new_module
            setattr(ultralytics.nn.modules, module_name, new_module)
    
    # Extended list of classes
    class_map = {
        'conv': ['Conv', 'DWConv', 'GhostConv', 'LightConv', 'Focus', 'GhostBottleneck', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'CrossConv', 'MixConv2d', 'AutoShape', 'DFL'],
        'block': ['C2f', 'Bottleneck', 'BottleneckCSP', 'C3', 'C3x', 'SPP', 'SPPF', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'DFL'],
        'head': ['Detect', 'Segment', 'Pose', 'Classify', 'DFL'],
        'transformer': ['TransformerBlock', 'TransformerLayer'],
        'activation': ['SiLU', 'Hardswish', 'LeakyReLU', 'Mish']
    }
    
    # Add the classes to their respective modules
    for module_name, classes in class_map.items():
        module = sys.modules.get(f'ultralytics.nn.modules.{module_name}')
        if module:
            for class_name in classes:
                # Create a dummy class that can be pickled
                class_obj = type(class_name, (), {})
                setattr(module, class_name, class_obj)
    
    # Add DFL to multiple modules to be safe
    for module_name in module_names:
        module = sys.modules.get(f'ultralytics.nn.modules.{module_name}')
        if module:
            setattr(module, 'DFL', type('DFL', (), {}))
    
    # Add base Module class
    if not hasattr(ultralytics.nn, 'Module'):
        ultralytics.nn.Module = type('Module', (), {})
    
    print("Ultralytics module structure patched for compatibility with extended classes")