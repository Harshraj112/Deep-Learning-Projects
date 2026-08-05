# Image Colorization with VGG16 Feature Extraction

This project trains a deep learning model to add color to black-and-white images. It uses the **LAB color space** and a **VGG16-based encoder** with an upsampling decoder to predict color channels from grayscale input.

## How It Works

1. **Color space conversion** – Images are converted from RGB to LAB color space instead of working directly in RGB.
   - `L` channel = lightness (the grayscale information)
   - `a`, `b` channels = the color information
2. **Input/Output split** – The `L` channel becomes the model's input (the black-and-white image), and the `a`/`b` channels become the target output the model learns to predict.
3. **Model architecture** – A pretrained **VGG16** (ImageNet weights) is used as a feature extractor. Its intermediate layers feed into a decoder built from `UpSampling2D` and `DepthwiseConv2D` layers that progressively reconstruct full-resolution `a`/`b` channel maps.
4. **Reconstruction** – At inference time, the predicted `a`/`b` channels are merged back with the original `L` channel and converted from LAB back to RGB/BGR to produce the final colorized image.

## Project Structure (notebook workflow)

| Step | Description |
|---|---|
| 1. Imports | Loads TensorFlow/Keras, OpenCV, scikit-image, PIL, and other supporting libraries |
| 2. Data preparation | Samples 500 images from a source folder, converts each to LAB, and separates `L` (lightness) and `a`/`b` (color) channels into arrays |
| 3. Data loading | Loads precomputed `lspace100.npy` and `abspace100.npy` arrays (cached preprocessed data) |
| 4. Model definition | Builds the VGG16 + decoder architecture |
| 5. Compilation | Compiles the model with the Adam optimizer and MAPE loss |
| 6. Preprocessing | Normalizes input/output arrays and splits data into train/test sets |
| 7. Training | Fits the model for 5 epochs |
| 8. Inference & evaluation | Generates predictions on train/test samples and rescales output to 0–255 |
| 9. Visualization | Merges predicted color channels with the lightness channel and displays the result with `matplotlib` |

## Requirements

- Python 3.11
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV (`cv2`)
- scikit-image (`skimage`)
- Pillow (`PIL`)
- Matplotlib

Install with:
```bash
pip install tensorflow numpy pandas opencv-python scikit-image pillow matplotlib
```

## Data

- The notebook expects a folder of source images (path currently hardcoded to a local Windows directory) to sample from for preprocessing.
- Preprocessed `L` and `a`/`b` arrays are expected as `lspace100.npy` and `abspace100.npy` in the working directory. These should be generated from your own image dataset before training.

## Usage

1. Update the image source folder path to point to your own dataset.
2. Run the preprocessing cell to generate `L`/`a`b` arrays (or supply your own `.npy` files).
3. Run the model definition, compilation, and training cells in order.
4. Run the inference cell to generate a colorized sample.
5. View the result with the final `plt.imshow(transfer)` cell.

## Known Issues / Things to Review

This notebook was captured as-is from an experimental workflow. A few things worth double-checking before relying on it:

- The hardcoded dataset path (`D:\Proj\vinita\colornet\`) is Windows-specific and will need to be changed for other environments.
- Nesting a full `Model` (the VGG16 sub-model) inside a `Sequential` model alongside a `Dense` layer is unusual and may need restructuring depending on your TensorFlow version.
- The `Adam(lr=...)` argument name is deprecated in newer TensorFlow/Keras versions (use `learning_rate=` instead).
- Train/test splitting logic (`test_inp=X[testsize:,]`) should be reviewed to ensure it produces the intended non-overlapping split.
- Only 5 training epochs are used, which is likely insufficient for a full convergence — treat this as a proof-of-concept rather than a production-ready model.
- GPU/CUDA was not detected in the environment that produced the saved outputs, so training ran on CPU.

## Output

The final cell visualizes one colorized test image by merging the predicted `a`/`b` channels with the original lightness channel and converting back to a viewable color image.