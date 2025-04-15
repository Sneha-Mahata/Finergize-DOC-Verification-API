import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import os

# Add the DetectionModel to the safe globals list
torch.serialization.add_safe_globals([DetectionModel])

# Import the Flask app after adding safe globals
from app import app

# This is necessary for local development
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))