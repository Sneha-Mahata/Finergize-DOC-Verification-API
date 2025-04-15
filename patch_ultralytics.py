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
    
    # Add common classes to each module
    common_classes = ['Conv', 'Sequential', 'Module', 'C2f', 'Bottleneck', 'SPPF', 'DFL']
    
    for module_name in module_names:
        module = sys.modules[f'ultralytics.nn.modules.{module_name}']
        for class_name in common_classes:
            # Create a dummy class that can be pickled
            class_obj = type(class_name, (), {})
            setattr(module, class_name, class_obj)
    
    # Explicitly add classes known to be needed
    conv_module = sys.modules['ultralytics.nn.modules.conv']
    setattr(conv_module, 'Conv', type('Conv', (), {}))
    
    block_module = sys.modules['ultralytics.nn.modules.block']
    setattr(block_module, 'C2f', type('C2f', (), {}))
    setattr(block_module, 'Bottleneck', type('Bottleneck', (), {}))
    
    print("Ultralytics module structure patched for compatibility")