function noisyImg = applyNoiseToImage(img, noiseToApply, noiseStep, varianceSchedule)
%APPLYNOISETOIMAGE Processo forward: x_t = sqrt(aBar_t)*x0 + sqrt(1-aBar_t)*eps.
%   img, noiseToApply: [H W C B] (dlarray "SSCB" o numerici).
%   noiseStep: vettore NUMERICO 1 x B (o scalare) di passi in 1..T.
if ~isa(img, "dlarray"),          img = dlarray(single(img), "SSCB"); end
if ~isa(noiseToApply, "dlarray"), noiseToApply = dlarray(single(noiseToApply), "SSCB"); end

alphaBar = cumprod(1 - varianceSchedule);
ab = reshape(alphaBar(noiseStep), 1, 1, 1, []);
ab = dlarray(single(ab), "SSCB");

noisyImg = sqrt(ab) .* img + sqrt(1 - ab) .* noiseToApply;
end
