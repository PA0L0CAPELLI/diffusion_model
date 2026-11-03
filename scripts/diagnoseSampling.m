%% Diagnostica del campionamento (usa il checkpoint gia' addestrato, nessun retraining)
P = projectPaths();
C = load(P.checkpoint);
D = load(P.patches, "XTest", "YTest", "classNames");
vs = C.varianceSchedule; T = numel(vs); ab = cumprod(1 - vs);
nets = {C.net, C.netEMA}; netNames = ["raw", "EMA"];

n = 48;
idx = randperm(size(D.XTest, 4), n);
X0 = toModelRange(D.XTest(:,:,:,idx));
y  = dlarray(single(double(D.YTest(idx))'), "CB");

%% 1) Errore di predizione del rumore per t (baseline "predico 0" = 1.0)
ts = [10 50 100 200 300 400 500];
res = zeros(numel(ts), 2);
for k = 1:numel(ts)
    t = ts(k) * ones(1, n);
    noise = randn(size(X0), "single");
    Xt = applyNoiseToImage(X0, noise, t, vs);
    for m = 1:2
        pred = predict(nets{m}, Xt, dlarray(single(t), "CB"), y);
        res(k, m) = double(extractdata(mean((pred - dlarray(noise, "SSCB")).^2, "all")));
    end
end
disp(array2table(res, VariableNames=netNames, RowNames=string(ts)))
if ~isfolder(P.results), mkdir(P.results); end
writetable(array2table([ts(:) res], VariableNames=["t" netNames]), fullfile(P.results, "noise_mse.csv"));

%% 2) Ricostruzione a un passo di x0 (rete EMA): originale / rumoroso / stima
for tShow = [50 200]
    t = tShow * ones(1, n);
    noise = randn(size(X0), "single");
    Xt = applyNoiseToImage(X0, noise, t, vs);
    pred = extractdata(predict(C.netEMA, Xt, dlarray(single(t), "CB"), y));
    x0hat = (extractdata(Xt) - sqrt(1 - ab(tShow)) * pred) / sqrt(ab(tShow));
    figure("Name", "Ricostruzione a un passo, t=" + tShow); tiledlayout(3, 6, TileSpacing="none");
    for r = 1:3
        for k = 1:6
            nexttile
            switch r
                case 1, im = X0(:,:,:,k);
                case 2, im = extractdata(Xt(:,:,:,k));
                case 3, im = x0hat(:,:,:,k);
            end
            imshow(toImageRange(im))
        end
    end
end

%% 3) Traiettoria di campionamento: dove esplode max|x|?
cls = repelem(1:3, 2);
cfgs = {"EMA, clip x0",     C.netEMA, true;
        "EMA, senza clip",  C.netEMA, false;
        "raw, senza clip",  C.net,    false};
figure("Name", "max|x| durante il campionamento"); hold on
for k = 1:size(cfgs, 1)
    [imgs, tr] = generateConditionalImages(cfgs{k,2}, vs, C.imageSize, cls, C.numClasses, 1, cfgs{k,3});
    semilogy(1:T, tr(:,2), DisplayName=cfgs{k,1});
    figure("Name", "Campioni: " + cfgs{k,1}); tiledlayout(1, 6, TileSpacing="none");
    for j = 1:6, nexttile; imshow(imgs(:,:,:,j)); end
end
figure(findobj("Name", "max|x| durante il campionamento")); set(gca, "YScale", "log", "XDir", "reverse");
xlabel("t (da 500 a 1)"); ylabel("max |x|"); legend; grid on
