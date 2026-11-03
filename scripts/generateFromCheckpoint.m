%% Generazione di immagini sintetiche per classe dal checkpoint addestrato
% Salva i PNG in generated/ e, per ogni valore di w, controlla se i campioni
% sono copie di patch di training (vicino piu' prossimo).
% Risultati numerici in results/memorization.csv.
P = projectPaths();
C = load(P.checkpoint);
D = load(P.patches, "XTrain", "classNames");
net = C.netEMA; vs = C.varianceSchedule;

numPerClass    = 8;
guidanceScales = [1 2];          % confronta l'effetto della guida (prova anche 3)
if ~isfolder(P.generated), mkdir(P.generated); end
if ~isfolder(P.results),   mkdir(P.results);   end
cls = repelem(1:C.numClasses, numPerClass);
memo = table();

for w = guidanceScales
    imgs = generateConditionalImages(net, vs, C.imageSize, cls, C.numClasses, w, true);
    files = strings(numel(cls), 1);
    figure("Name", "w = " + w);
    tiledlayout(C.numClasses, numPerClass, TileSpacing="none", Padding="compact");
    for k = 1:numel(cls)
        nexttile; imshow(imgs(:,:,:,k));
        if mod(k-1, numPerClass) == 0, ylabel(C.classNames(cls(k))); end
        files(k) = sprintf("%s_w%.1f_%02d.png", C.classNames(cls(k)), w, k);
        imwrite(imgs(:,:,:,k), fullfile(P.generated, files(k)));
    end
    drawnow

    % Controllo memorizzazione: patch di training piu' vicina a ogni campione
    [rmse, bestIdx] = nearestTrainPatch(imgs, D.XTrain);
    fprintf("w=%g | RMSE al vicino piu' prossimo: min %.3f | mediana %.3f (copie esatte: < ~0.02)\n", ...
        w, min(rmse), median(rmse));
    memo = [memo; table(reshape(C.classNames(cls), [], 1), repmat(w, numel(cls), 1), files, ...
        rmse(:), bestIdx(:), VariableNames=["Classe", "w", "File", "RMSE_NN", "IndiceTrainNN"])]; %#ok<AGROW>

    figure("Name", "w = " + w + ": generato (sopra) vs vicino piu' prossimo nel train (sotto)");
    tiledlayout(2, numel(cls), TileSpacing="none", Padding="compact");
    for k = 1:numel(cls), nexttile; imshow(imgs(:,:,:,k)); end
    for k = 1:numel(cls), nexttile; imshow(D.XTrain(:,:,:,bestIdx(k))); end
end
writetable(memo, fullfile(P.results, "memorization.csv"));

%% ------------------------- funzioni locali -------------------------
function [rmse, bestIdx] = nearestTrainPatch(imgs, XTrain)
% Per ogni immagine generata trova la patch di training piu' vicina (distanza
% L2), calcolata a blocchi per non saturare la memoria.
% imgs: [S S 3 N] in [0,1].  XTrain: [S S 3 M] uint8.
N = size(imgs, 4);
G = reshape(imgs, [], N);                                  % pixel x N
M = size(XTrain, 4); chunk = 4000;
best = inf(1, N); bestIdx = ones(1, N);
for s = 1:chunk:M
    e = min(s + chunk - 1, M);
    X = reshape(single(XTrain(:,:,:,s:e)), [], e - s + 1) / 255;
    d = sum(G.^2, 1)' + sum(X.^2, 1) - 2 * (G' * X);      % N x chunk (L2 al quadrato)
    [m, i] = min(d, [], 2); m = m'; i = i';
    upd = m < best; best(upd) = m(upd); bestIdx(upd) = s - 1 + i(upd);
end
rmse = sqrt(max(best, 0) / size(G, 1));                    % errore RMS per pixel in [0,1]
end
