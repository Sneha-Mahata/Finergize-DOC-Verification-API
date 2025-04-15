import os
import torch

# Override torch.load to use weights_only=False
original_torch_load = torch.load
def custom_torch_load(*args, **kwargs):
    # Force weights_only to False
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Replace torch.load with our custom version
torch.load = custom_torch_load

# Import the Flask app after modifying torch.load
from app import app

# This is necessary for local development
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))