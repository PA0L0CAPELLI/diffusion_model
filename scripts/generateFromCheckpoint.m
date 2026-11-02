%% Generazione di immagini sintetiche per classe dal checkpoint addestrato
% Salva PNG in "generated/" e controlla se i campioni sono copie di patch di training.
C = load("diffusionLSCCheckpoint.mat");
D = load("LSCPatches.mat", "XTrain", "classNames");
net = C.netEMA; vs = C.varianceSchedule;

numPerClass    = 8;
guidanceScales = [1 2];          % confronta l'effetto della guida (prova anche 3)
outDir = "generated"; if ~isfolder(outDir), mkdir(outDir); end
cls = repelem(1:C.numClasses, numPerClass);

for w = guidanceScales
    imgs = generateConditionalImages(net, vs, C.imageSize, cls, C.numClasses, w, true);
    figure("Name", "w = " + w);
    tiledlayout(C.numClasses, numPerClass, TileSpacing="none", Padding="compact");
    for k = 1:numel(cls)
        nexttile; imshow(imgs(:,:,:,k));
        if mod(k-1, numPerClass) == 0, ylabel(C.classNames(cls(k))); end
        imwrite(imgs(:,:,:,k), fullfile(outDir, sprintf("%s_w%.1f_%02d.png", C.classNames(cls(k)), w, k)));
    end
    drawnow
end

%% Controllo memorizzazione sull'ultimo gruppo (vicino piu' prossimo nel train)
G = reshape(imgs, [], numel(cls));                        % pixel x N, in [0,1]
M = size(D.XTrain, 4); chunk = 4000;
best = inf(1, numel(cls)); bestIdx = ones(1, numel(cls));
for s = 1:chunk:M
    e = min(s + chunk - 1, M);
    P = reshape(single(D.XTrain(:,:,:,s:e)), [], e - s + 1) / 255;
    d = sum(G.^2, 1)' + sum(P.^2, 1) - 2 * (G' * P);      % N x chunk (distanza L2 al quadrato)
    [m, i] = min(d, [], 2); m = m'; i = i';
    upd = m < best; best(upd) = m(upd); bestIdx(upd) = s - 1 + i(upd);
end
rmse = sqrt(max(best, 0) / size(G, 1));                   % errore RMS per pixel in [0,1]
fprintf("RMSE al vicino piu' prossimo: min %.3f | mediana %.3f (copie esatte: < ~0.02)\n", min(rmse), median(rmse));

figure("Name", "Generato (sopra) vs vicino piu' prossimo nel train (sotto)");
tiledlayout(2, numel(cls), TileSpacing="none", Padding="compact");
for k = 1:numel(cls), nexttile; imshow(imgs(:,:,:,k)); end
for k = 1:numel(cls), nexttile; imshow(D.XTrain(:,:,:,bestIdx(k))); end
