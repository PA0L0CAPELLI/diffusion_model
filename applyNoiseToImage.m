function noisyImg = applyNoiseToImage(img, noiseToApply, noiseStep, varianceSchedule)
%APPLYNOISETOIMAGE Apply noiseStep steps of the forward diffusion process.
%   Reproduced from the MathWorks "Generate Images Using Diffusion"
%   example: x_t = sqrt(alphaBar_t)*x_0 + sqrt(1 - alphaBar_t)*epsilon.

alphaBar = cumprod(1 - varianceSchedule);
alphaBarT = dlarray(alphaBar(noiseStep), "CBSS");

noisyImg = sqrt(alphaBarT) .* img + sqrt(1 - alphaBarT) .* noiseToApply;
end
