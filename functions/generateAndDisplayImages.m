function images = generateAndDisplayImages(net, varianceSchedule, imageSize, numImages, numChannels, displayFrequency)
%GENERATEANDDISPLAYIMAGES Reverse the diffusion process to sample images.
%   Reproduced from the MathWorks "Generate Images Using Diffusion"
%   example. Starts from Gaussian noise and iteratively denoises using
%   the trained network's noise predictions.

images = randn([imageSize numChannels numImages]);

alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule .* (1 - alphaBarPrev) ./ (1 - alphaBar);

numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    if noiseStep ~= 1
        z = randn([imageSize, numChannels, numImages]);
    else
        z = zeros([imageSize, numChannels, numImages]);
    end

    predictedNoise = predict(net, images, noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep)) * z;
    predNoise = varianceSchedule(noiseStep) * predictedNoise / sqrt(1 - alphaBar(noiseStep));

    images = 1/sqrtOneMinusBeta * (images - predNoise) + addedNoise;

    if mod(noiseStep, displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay, "t = " + noiseStep)
        for ii = 1:numImages
            nexttile
            imshow(rescale(images(:,:,:,ii)));
        end
        drawnow
    end
end

tLay = tiledlayout("flow");
title(tLay, "t = 0")
for ii = 1:numImages
    nexttile
    imshow(rescale(images(:,:,:,ii)));
end
end
