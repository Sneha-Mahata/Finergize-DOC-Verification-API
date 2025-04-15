import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import app

# Add the DetectionModel to the safe globals list
torch.serialization.add_safe_globals([DetectionModel])

# This is necessary for the Render deployment
if __name__ == "__main__":
    app.app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))