# CIFAR-10 Image Classifier

A small computer vision project that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.

The project includes:

- A reproducible Nix development environment
- CPU-based model training
- A saved PyTorch model (`.pt`)
- ONNX export for browser inference
- A React + TypeScript frontend for uploading images and viewing predictions

## What the Model Can Classify

The model can classify an image into one of the 10 CIFAR-10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

> The model will always choose one of these 10 classes. Images outside these categories may still receive a prediction and confidence score.

## Dataset

This project uses the **CIFAR-10** dataset.

CIFAR-10 contains:

- 50,000 training images
- 10,000 test images
- 10 classes
- 32 × 32 RGB images

Dataset source: https://www.kaggle.com/c/cifar-10

The dataset is downloaded automatically through `torchvision` when the training script is run.

## Requirements

The project is designed to run inside the included Nix development environment.

You will need:

- Nix
- Git
- Node.js
- npm

The Nix development environment provides the Python, PyTorch, ONNX, and Node.js dependencies needed for the project.

If you are using Windows, run the project through **WSL2 / Ubuntu**.

The frontend is built with:

- React
- TypeScript
- Vite
- ONNX Runtime Web

## 1. Clone the Repository

```bash
git clone https://github.com/ijf03/cifar10-image-classifier.git
cd cifar10-image-classifier
```

## 2. Enter the Nix Development Environment

Run:

```bash
nix develop
```

Once inside the environment, verify Python and PyTorch with:

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

## 3. Train the Model

Run:

```bash
python train.py
```

The training script will:

1. Download/check the CIFAR-10 dataset
2. Prepare the training and test images
3. Train the CNN on CPU
4. Print loss and training accuracy after each epoch
5. Evaluate the model on the test dataset
6. Print the final test accuracy
7. Save the trained model

The saved PyTorch model is:

```text
cifar10_cnn.pt
```

## 4. Export the Model to ONNX

Make sure `cifar10_cnn.pt` exists first, then run:

```bash
python export_onnx.py
```

The exporter places the ONNX model in:

```text
frontend/public/cifar10_cnn.onnx
```

The model should be exported as a single ONNX file using `external_data=False` so it can be loaded directly in the browser.

## 5. Run the Frontend

Make sure you are still inside the Nix development environment.

Move into the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Vite will display a local address similar to:

```text
http://localhost:5173
```

Open that address in your browser.

## Using the Web Interface

1. Click **Choose an image**.
2. Select a JPG, PNG, or other supported image.
3. The image is resized to 32 × 32 and normalized.
4. ONNX Runtime Web runs the exported model in the browser.
5. The page shows the predicted CIFAR-10 class.
6. The page also shows the model's confidence score.

## How the Application Works

```text
Upload image
    ↓
Resize to 32 × 32 RGB
    ↓
Normalize image values
    ↓
Convert image to tensor
    ↓
ONNX Runtime Web
    ↓
CNN produces 10 class scores
    ↓
Highest score becomes the prediction
```

The frontend runs the exported ONNX model directly in the browser, so a separate Python backend is not required for prediction.

## Model Architecture

```text
Input: 3 × 32 × 32 RGB image
    ↓
Conv2D: 3 → 32 channels
Batch Normalization
ReLU
Max Pooling
    ↓
Conv2D: 32 → 64 channels
Batch Normalization
ReLU
Max Pooling
    ↓
Flatten
    ↓
Linear: 4096 → 128
ReLU
    ↓
Linear: 128 → 10
    ↓
Prediction
```

Training uses:

- PyTorch
- Cross Entropy Loss
- AdamW optimizer
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Cosine learning-rate scheduler
- Batch size: `64`
- CPU training


## Troubleshooting

### `nix: command not found`

Nix is not installed or has not been loaded into the current terminal session.

Check with:

```bash
nix --version
```

### Nix build uses too much memory

Use:

```bash
nix develop --max-jobs 1 --cores 2
```

This reduces parallel compilation.

### Frontend cannot classify an image

Check that the ONNX model exists:

```bash
ls frontend/public/cifar10_cnn.onnx
```

If it does not exist, export the model again:

```bash
python export_onnx.py
```

### ONNX Runtime cannot load an external `.data` file

Make sure the ONNX export uses:

```python
external_data=False
```

This keeps the model weights inside one `.onnx` file.

## Notes

This project was created as a small machine learning exercise focused on:

- Understanding how a CNN works
- Learning the PyTorch training workflow
- Experimenting with model parameters
- Verifying model behaviour
