function [images, trace] = generateConditionalImages(net, varianceSchedule, imageSize, classIdx, numClasses, guidanceScale, clipX0)
%GENERATECONDITIONALIMAGES Campionamento DDPM con classifier-free guidance.
%   classIdx: classe (1..numClasses) di ciascuna immagine da generare.
%   guidanceScale: 1 = solo condizionato; >1 = guida piu' forte.
%   clipX0 (default true): stima x0, la limita a [-1,1] e poi calcola la media
%     del posteriore. Matematicamente equivale alla forma classica quando
%     x0 e' in range, ma impedisce ai valori di esplodere durante il campionamento.
%   images: [S S 3 N] single in [0,1].   trace: [T x 2] = (std, max|x|) per passo.
if nargin < 6, guidanceScale = 1; end
if nargin < 7, clipX0 = true; end

N = numel(classIdx);
T = numel(varianceSchedule);
beta = varianceSchedule;
alpha = 1 - beta;
alphaBar = cumprod(alpha);
alphaBarPrev = [1 alphaBar(1:end-1)];
postVar = beta .* (1 - alphaBarPrev) ./ (1 - alphaBar);

x = randn([imageSize imageSize 3 N], "single");
c     = dlarray(single(classIdx(:)'), "CB");
cNull = dlarray(single((numClasses + 1) * ones(1, N)), "CB");
trace = zeros(T, 2);

for t = T:-1:1
    tt = dlarray(single(t * ones(1, N)), "CB");
    xin = dlarray(x, "SSCB");

    epsC = predict(net, xin, tt, c);
    if guidanceScale ~= 1
        epsU = predict(net, xin, tt, cNull);
        epsHat = epsU + guidanceScale * (epsC - epsU);
    else
        epsHat = epsC;
    end
    epsHat = extractdata(epsHat);

    if clipX0
        x0 = (x - sqrt(1 - alphaBar(t)) * epsHat) / sqrt(alphaBar(t));
        x0 = min(max(x0, -1), 1);
        c1 = sqrt(alphaBarPrev(t)) * beta(t) / (1 - alphaBar(t));
        c2 = sqrt(alpha(t)) * (1 - alphaBarPrev(t)) / (1 - alphaBar(t));
        x = c1 * x0 + c2 * x;
    else
        x = (x - beta(t) / sqrt(1 - alphaBar(t)) * epsHat) / sqrt(alpha(t));
    end
    if t > 1
        x = x + sqrt(postVar(t)) * randn(size(x), "single");
    end
    trace(t, :) = [std(x(:)), max(abs(x(:)))];
end
images = (min(max(x, -1), 1) + 1) / 2;
end