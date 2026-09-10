import torch

from train import make_model


checkpoint = torch.load("cifar10_cnn.pt", map_location="cpu")

model = make_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    (dummy_input,),
    "cifar10_cnn.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    dynamo=True,
)

print("Exported model to cifar10_cnn.onnx")