# `trainDiffusion.m` Explained

This file trains a **class-conditional diffusion model** on histology image patches.

The goal is simple:
- learn from real patches,
- learn how to remove noise step by step,
- generate new synthetic patches that match a requested class,
- save the model so training can be resumed later.

---

## 1. What a Diffusion Model Does

A diffusion model learns image generation in two phases:

1. **Forward process**  
   Start with a real image and gradually add noise until it becomes almost random.

2. **Reverse process**  
   Teach a neural network to remove that noise step by step.

After training, the model can start from pure noise and turn it into a realistic image.

In this project, the model is also **class-conditional**, meaning it can generate images for a chosen class:
- CLL
- FL
- MCL

---

## 2. Script Overview

`trainDiffusion.m` does five main things:

1. loads the dataset,
2. defines the training settings,
3. checks that the diffusion noise process looks correct,
4. trains the diffusion network,
5. saves checkpoints and generates sample images.

---

## 3. Configuration Section

This section defines the main training settings.

### Data and model size

- `dataFile = "LSCPatches.mat"`
  - File containing the patch dataset.

- `imageSize = 64`
  - Each patch is 64 by 64 pixels.

- `numClasses = 3`
  - There are three classes in the dataset.

- `baseChannels = 32`
  - Controls the size of the neural network.
  - Larger values usually mean a bigger and slower model.

- `numNoiseSteps = 500`
  - Number of diffusion steps used in the noising process.

### Noise schedule

The script creates `varianceSchedule`, which controls how much noise is added at each step.

It uses a **cosine schedule** instead of a linear one because it often gives better diffusion behavior, especially when images are small.

The key idea is:
- early timesteps add little noise,
- later timesteps add more noise,
- the final timestep should leave almost pure noise.

### Training settings

- `miniBatchSize = 16`
  - Number of images processed in one update.

- `numIterations = 20000`
  - Total training steps.

- `learnRate = 2e-4`
  - How fast the network learns.

- `pUncond = 0.1`
  - Probability of removing the class label during training.
  - This supports **classifier-free guidance**.

- `emaDecay = 0.999`
  - Smoothing factor for the exponential moving average network.

### Monitoring and saving

- `evalEvery = 500`
  - How often test loss is computed.

- `sampleEvery = 5000`
  - How often sample images are generated during training.

- `checkpointEvery = 1000`
  - How often the model is saved.

- `guidanceScale = 1`
  - Controls how strongly the class label influences generation.

- `resume = false`
  - If `true`, training resumes from a saved checkpoint.
  - If `false`, training starts from scratch.

---

## 4. Data Loading Section

This section loads the dataset from `LSCPatches.mat`.

### Loaded variables

- `XTrain`
  - Training images.

- `YTrain`
  - Training labels.

- `XTest`
  - Test images.

- `YTest`
  - Test labels.

- `classNames`
  - Human-readable class names.

### Why this matters

The diffusion model needs:
- real images to learn from,
- labels to condition generation,
- test data to verify that the model is generalizing.

### Size check

The script prints the number of training and test patches, then checks that the patch size matches `imageSize`.

This prevents shape mismatches later in training.

---

## 5. Forward Diffusion Sanity Check

This section does not train the model. It only checks that the noise process behaves correctly.

### What it does

1. Computes `alphaBar`, which measures how much image signal remains over time.
2. Takes one real patch.
3. Adds increasing noise at different timesteps.
4. Displays the results.

### Why this is useful

A diffusion model depends on the forward noising process being correct.

You should see:
- a clean image at `t = 0`,
- progressively noisier images as `t` increases,
- almost pure noise near the final step.

If this does not happen, the diffusion setup is wrong.

---

## 6. Network Setup Section

This section creates the model or reloads a previous one.

### Resume training

If:
- `resume = true`
- and `diffusionLSCCheckpoint.mat` exists

then the script loads:
- `net`
- `netEMA`
- `iteration`

This lets training continue from where it stopped.

### Start fresh

If not resuming, the script creates a new network with:

```matlab
createDiffusionNetwork(imageSize, numClasses, baseChannels)
```

Then it initializes:
- `netEMA = net`
- `iteration = 0`

### What is `netEMA`?

`netEMA` is an **exponential moving average** version of the network.

It changes more slowly than the main network and is usually better for sampling images.

---

## 7. Training Loop Section

This is the core of the script.

Each iteration trains the model on a random batch of image patches.

### Step 1: Select a mini-batch

The script randomly chooses `miniBatchSize` images from the training set.

### Step 2: Apply augmentation

The batch is passed to `augmentBatch`, which applies:
- random rotation,
- random horizontal flip.

This makes the model less sensitive to orientation.

### Step 3: Normalize the images

The images are converted from `uint8` in `[0, 255]` to `single` values around `[-1, 1]`.

This is a common preprocessing step for neural networks.

### Step 4: Prepare the labels

The training labels are loaded from `YTrain`.

Then, for some samples, the label is replaced with a special null label:
- this happens with probability `pUncond`.

This trains the network to work both:
- with class information,
- without class information.

That ability is needed for classifier-free guidance.

### Step 5: Add noise

The script:
- creates random Gaussian noise,
- picks a random timestep `t`,
- corrupts the image using `applyNoiseToImage`.

At this point, the input is a noisy image, not a clean one.

### Step 6: Compute the loss

The function `modelLoss` compares:
- the noise predicted by the network,
- the true noise that was added.

If the prediction is close, the loss is small.

### Step 7: Update the weights

The network is updated with `adamupdate`.

This changes the model so it gets better at predicting noise in future iterations.

### Step 8: Update the EMA network

The script updates `netEMA` using a moving average of the current weights.

This produces a smoother model for generation.

### Step 9: Update the progress monitor

The monitor displays:
- training loss,
- test loss,
- iteration number,
- time per iteration.

---

## 8. Periodic Evaluation and Saving

Inside the training loop, the script performs extra tasks at fixed intervals.

### Every `evalEvery` iterations

The function `evaluateTestLoss` is called.

This measures loss on unseen test data.

Why this matters:
- training loss shows how well the model fits the training set,
- test loss shows whether it generalizes.

### Every `sampleEvery` iterations

The script calls `showSamples`.

This generates synthetic images and displays them in a figure.

This is important because diffusion models are often judged by visual quality, not only by loss.

### Every `checkpointEvery` iterations

The script saves a checkpoint to:

- `diffusionLSCCheckpoint.mat`

This file contains:
- `net`
- `netEMA`
- `varianceSchedule`
- `iteration`
- `classNames`
- `imageSize`
- `numClasses`

This makes it possible to resume training later.

---

## 9. Final Sampling Section

After training ends, the script generates 8 images per class.

This is the final qualitative test.

If training went well, the generated images should:
- look realistic,
- reflect the requested class,
- not look like random noise.

---

## 10. Local Functions

The script defines three helper functions at the end.

### `augmentBatch(batch)`

Applies random spatial augmentation to each image:
- rotation,
- horizontal flip.

#### Why it is used
Histology patches usually do not have a fixed "correct" orientation, so these transformations are safe.

---

### `evaluateTestLoss(net, XTest, YTest, numClasses, varianceSchedule, n)`

Measures how well the network predicts noise on unseen test images.

#### What it does
1. Selects a random subset of test images.
2. Normalizes them.
3. Adds noise.
4. Predicts the noise with the network.
5. Computes mean squared error.

#### Why it matters
It helps detect overfitting.

---

### `showSamples(net, varianceSchedule, imageSize, numClasses, classNames, nPer, w)`

Generates and displays synthetic images for each class.

#### What the arguments mean
- `nPer`
  - Number of generated images per class.

- `w`
  - Guidance strength during generation.

#### Why it matters
This gives a direct visual check of the model output.

---

## 11. External Functions Used

These functions are not defined in this file, but the script depends on them.

### `createDiffusionNetwork(...)`
Builds the diffusion model architecture.

### `applyNoiseToImage(...)`
Implements the forward process: it adds noise to clean images.

### `modelLoss(...)`
Computes the training loss by comparing predicted noise with true noise.

### `generateConditionalImages(...)`
Runs the reverse diffusion process and creates synthetic images.

---

## 12. Glossary

### Adam
An optimization algorithm used to update neural network weights during training.

### Batch
A small group of images processed together in one training step.

### Class-conditional
A model that uses class labels to control what kind of image it generates.

### Checkpoint
A saved version of the model that can be loaded later to continue training.

### Diffusion model
A generative model that learns to turn noise into images by reversing a noising process.

### EMA
Short for **Exponential Moving Average**. A smooth version of the model weights, often used for sampling.

### Forward process
The process of gradually adding noise to an image.

### Guidance scale
A value that controls how strongly the class label affects image generation.

### Loss
A number that tells how wrong the model is. Lower is better.

### Mini-batch
A small subset of the dataset used in one update step.

### Noise schedule
A rule that defines how much noise is added at each diffusion step.

### Reverse process
The learned process of removing noise step by step to create an image.

### Sampling
Using the trained model to generate new images.

### Test loss
Loss measured on data not used for training.

### Training loss
Loss measured on data used during training.

### Unconditional generation
Generating images without giving the model a class label.

### Classifier-free guidance
A technique that trains the model both with and without labels, so generation can be guided later.

---
