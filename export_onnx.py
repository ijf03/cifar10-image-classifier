import torch

from train import make_model
from pathlib import Path


checkpoint = torch.load("cifar10_cnn.pt", map_location="cpu")

model = make_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)

output_path = Path("frontend/public/cifar10_cnn.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)


torch.onnx.export(
    model,
    (dummy_input,),
    output_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    dynamo=True,
    external_data=False,
)

print(f"Exported model to {output_path}")