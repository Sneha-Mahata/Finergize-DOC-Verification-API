import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
import os

# Add both classes to the safe globals list
torch.serialization.add_safe_globals([DetectionModel, Sequential])

# Import the Flask app after adding safe globals
from app import app

# This is necessary for local development
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))