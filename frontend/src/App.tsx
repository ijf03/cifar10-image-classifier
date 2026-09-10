import { useState } from "react";
import * as ort from "onnxruntime-web";
import "./App.css";

const classes = [
  "airplane",
  "automobile",
  "bird",
  "cat",
  "deer",
  "dog",
  "frog",
  "horse",
  "ship",
  "truck",
];

const mean = [0.4914, 0.4822, 0.4465];
const std = [0.2470, 0.2435, 0.2616];

type ClassResult = {
  name: string;
  confidence: number;
};

let session: ort.InferenceSession | null = null;

async function getSession() {
  if (!session) {
    console.log("Loading ONNX model...");

    session = await ort.InferenceSession.create(
      "/cifar10_cnn.onnx",
      {
        executionProviders: ["wasm"],
      }
    );

    console.log("Model loaded");
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);
  }

  return session;
}

function softmax(values: number[]) {
  const max = Math.max(...values);

  const exponentials = values.map((value) =>
    Math.exp(value - max)
  );

  const total = exponentials.reduce(
    (sum, value) => sum + value,
    0
  );

  return exponentials.map((value) => value / total);
}

async function imageToTensor(file: File) {
  const image = new Image();
  const imageUrl = URL.createObjectURL(file);

  image.src = imageUrl;

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () =>
      reject(new Error("The uploaded image could not be read."));
  });

  const canvas = document.createElement("canvas");

  canvas.width = 32;
  canvas.height = 32;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Could not create image canvas.");
  }

  /*
   * Centre crop the uploaded image into a square.
   * This avoids stretching a wide/tall image.
   */
  const sourceSize = Math.min(
    image.naturalWidth,
    image.naturalHeight
  );

  const sourceX =
    (image.naturalWidth - sourceSize) / 2;

  const sourceY =
    (image.naturalHeight - sourceSize) / 2;

  context.drawImage(
    image,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    32,
    32
  );

  const pixels = context.getImageData(
    0,
    0,
    32,
    32
  ).data;

  const tensorData =
    new Float32Array(3 * 32 * 32);

  for (let i = 0; i < 32 * 32; i++) {
    const red = pixels[i * 4] / 255;
    const green = pixels[i * 4 + 1] / 255;
    const blue = pixels[i * 4 + 2] / 255;

    tensorData[i] =
      (red - mean[0]) / std[0];

    tensorData[1024 + i] =
      (green - mean[1]) / std[1];

    tensorData[2048 + i] =
      (blue - mean[2]) / std[2];
  }

  URL.revokeObjectURL(imageUrl);

  return new ort.Tensor(
    "float32",
    tensorData,
    [1, 3, 32, 32]
  );
}

function App() {
  const [preview, setPreview] =
    useState<string | null>(null);

  const [results, setResults] =
    useState<ClassResult[]>([]);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  async function handleImage(file: File) {
    const previewUrl = URL.createObjectURL(file);

    setPreview(previewUrl);
    setResults([]);
    setError(null);
    setLoading(true);

    try {
      const model = await getSession();

      const tensor =
        await imageToTensor(file);

      const inputName =
        model.inputNames[0];

      console.log(
        "Input:",
        inputName,
        tensor.dims
      );

      const output = await model.run({
        [inputName]: tensor,
      });

      const outputName =
        model.outputNames[0];

      const scores = Array.from(
        output[outputName]
          .data as Float32Array
      );

      console.log("Raw scores:", scores);

      const probabilities =
        softmax(scores);

      const classResults =
        probabilities
          .map((confidence, index) => ({
            name: classes[index],
            confidence: confidence * 100,
          }))
          .sort(
            (a, b) =>
              b.confidence - a.confidence
          );

      setResults(classResults);
    } catch (error) {
      console.error(
        "Classification error:",
        error
      );

      setError(
        error instanceof Error
          ? error.message
          : "Could not classify this image."
      );
    } finally {
      setLoading(false);
    }
  }

  const prediction = results[0];

  return (
    <main className="page">
      <div className="container">

        <header className="hero">
          <p className="eyebrow">
            COMPUTER VISION PROJECT
          </p>

          <h1>
            CIFAR-10
            <br />
            Image Classifier
          </h1>

          <p className="hero-description">
            A small convolutional neural
            network trained with PyTorch on
            the CIFAR-10 dataset. The trained
            model was exported to ONNX and
            runs directly in your browser.
          </p>

          <div className="tech">
            <span>PyTorch</span>
            <span>ONNX</span>
            <span>React</span>
            <span>TypeScript</span>
          </div>
        </header>

        <section className="classifier">
          <div className="upload-panel">
            <div className="section-heading">
              <span>01</span>

              <div>
                <h2>Upload an image</h2>

                <p>
                  Choose an image and the model
                  will try to classify it.
                </p>
              </div>
            </div>

            <label className="upload-box">

              {preview ? (
                <img
                  src={preview}
                  alt="Uploaded preview"
                />
              ) : (
                <div className="upload-placeholder">
                  <div className="upload-icon">
                    ↑
                  </div>

                  <strong>
                    Choose an image
                  </strong>

                  <span>
                    JPG, PNG or WEBP
                  </span>
                </div>
              )}

              <input
                type="file"
                accept="image/*"
                onChange={(event) => {
                  const file =
                    event.target.files?.[0];

                  if (file) {
                    handleImage(file);
                  }
                }}
              />
            </label>

            {preview && (
              <p className="replace">
                Click the image to choose
                another file
              </p>
            )}
          </div>

          <div className="result-panel">
            <div className="section-heading">
              <span>02</span>

              <div>
                <h2>Prediction</h2>

                <p>
                  The model's highest scoring
                  categories.
                </p>
              </div>
            </div>

            {!prediction &&
              !loading &&
              !error && (
                <div className="empty-result">
                  <p>
                    Upload an image to see a
                    prediction.
                  </p>
                </div>
              )}

            {loading && (
              <div className="empty-result">
                <div className="loader" />
                <p>Running model...</p>
              </div>
            )}

            {error && (
              <div className="error">
                <strong>
                  Classification failed
                </strong>

                <p>{error}</p>
              </div>
            )}

            {prediction &&
              !loading && (
                <div className="prediction">

                  <p className="prediction-label">
                    MODEL PREDICTION
                  </p>

                  <h3>
                    {prediction.name}
                  </h3>

                  <div className="confidence-number">
                    {prediction.confidence.toFixed(
                      1
                    )}
                    <span>% confidence</span>
                  </div>

                  <div className="confidence-list">
                    {results
                      .slice(0, 3)
                      .map((result) => (
                        <div
                          className="confidence-item"
                          key={result.name}
                        >
                          <div className="confidence-top">
                            <span>
                              {result.name}
                            </span>

                            <span>
                              {result.confidence.toFixed(
                                1
                              )}
                              %
                            </span>
                          </div>

                          <div className="bar">
                            <div
                              className="bar-fill"
                              style={{
                                width: `${result.confidence}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>

                  <p className="confidence-note">
                    Confidence represents the
                    model's relative score, not
                    a guarantee that the
                    prediction is correct.
                  </p>
                </div>
              )}
          </div>
        </section>

        <section className="categories">
          <div className="section-heading">
            <span>03</span>

            <div>
              <h2>
                What can it recognise?
              </h2>

              <p>
                The model was trained to choose
                between these 10 CIFAR-10
                classes.
              </p>
            </div>
          </div>

          <div className="category-grid">
            {classes.map(
              (className, index) => (
                <div
                  className="category"
                  key={className}
                >
                  <span>
                    {String(index + 1).padStart(
                      2,
                      "0"
                    )}
                  </span>

                  {className}
                </div>
              )
            )}
          </div>
        </section>

        <section className="about">
          <div className="section-heading">
            <span>04</span>

            <div>
              <h2>About the project</h2>
            </div>
          </div>

          <div className="about-grid">
            <p>
              A small CNN trained
              using CIFAR-10 while learning how
              image tensors, convolution
              layers, loss, backpropagation and
              model evaluation work.
            </p>

            <a href="https://github.com/ijf03/cifar10-image-classifier" target="_blank" rel="noopener noreferrer">
              <strong>
                View the source code on GitHub
              </strong>
            </a>

          
          </div>
        </section>

      </div>
    </main>
  );
}

export default App;