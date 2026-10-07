%% Load Data
imds = loadDatastore("idx","/Users/paolocapelli/Desktop/MNIST");
imgSize = [32 32];
audsImds = augmentedImageDatastore(imgSize, imds);
%% Forward Diffusion Process
numNoiseSteps = 500;
betaMin = 0.0001;
betaMax = 0.02;
varianceSchedule = linspace(betaMin, betaMax, numNoiseSteps);

% Sanity-check the forward process on one training image.
img = read(audsImds);
img = img{1,1};
img = img{:};
img = rescale(img, -1, 1);

figure
tiledlayout("flow");
nexttile
imshow(img, [])
title("t = 0");
for i = 1:5
    nexttile
    noise = randn(size(img), like=img);
    noiseStepsToApply = numNoiseSteps/5 * i;
    noisyImg = applyNoiseToImage(img, noise, noiseStepsToApply, varianceSchedule);
    noisyImg = extractdata(noisyImg);
    imshow(noisyImg, [])
    title("t = " + string(noiseStepsToApply));
end

%% Define Network
numInputChannels = 1;
net = createDiffusionNetwork(numInputChannels);
disp(net)

%% Specify Training Options
miniBatchSize = 128;
numEpochs = 50;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

%% Train Model
doTraining = true;   % set false once you have a saved net to reload

mbq = minibatchqueue(audsImds, 1, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="discard");

averageGrad = [];
averageSqGrad = [];

numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

if doTraining
    monitor = trainingProgressMonitor(...
        Metrics="Loss", ...
        Info=["Epoch","Iteration"], ...
        XLabel="Iteration");

    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            img = next(mbq);

            targetNoise = randn(size(img), Like=img);
            noiseStep = dlarray(randi(numNoiseSteps,[1 miniBatchSize],Like=img), "CB");
            noisyImage = applyNoiseToImage(img, targetNoise, noiseStep, varianceSchedule);

            [loss, gradients, state] = dlfeval(@modelLoss, net, noisyImage, noiseStep, targetNoise);
            net.State = state;
            gradients = dlupdate(@(g) max(min(g, 1), -1), gradients);

            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, iteration, ...
                learnRate, gradientDecayFactor, squaredGradientDecayFactor);

            recordMetrics(monitor, iteration, Loss=loss);
            updateInfo(monitor, Epoch=epoch, Iteration=iteration);
            monitor.Progress = 100 * iteration/numIterations;
        end

        % Preview generated digits at the end of each epoch
        numImages = 16;
        displayFrequency = numNoiseSteps + 1;   % only show the final result
        generateAndDisplayImages(net, varianceSchedule, imgSize, numImages, numInputChannels, displayFrequency);

        save("diffusionMnistCheckpoint.mat", "net", "varianceSchedule", "-v7.3");
        pause(0.1);
    end
else
    load("diffusionMnistCheckpoint.mat", "net", "varianceSchedule");
end

%% Generate New Images
numImages = 9;
displayFrequency = 10;
figure
generatedImages = generateAndDisplayImages(net, varianceSchedule, imgSize, numImages, numInputChannels, displayFrequency);