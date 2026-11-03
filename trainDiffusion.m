%% Class-conditional diffusion su LSC (CLL / FL / MCL) - training
% Prima di tutto:  P = projectPaths(); prepareLSCDataset(fullfile(P.data, "LSC"))  -> crea data/LSCPatches.mat
% Per un test rapido: numIterations = 20 e controlla il tempo/iterazione.

%% Configurazione
P             = projectPaths();
dataFile      = P.patches;          % data/LSCPatches.mat
ckptFile      = P.checkpoint;       % checkpoints/diffusionLSCCheckpoint.mat
imageSize     = 64;                 % deve coincidere con OutputSize di prepareLSCDataset
numClasses    = 3;
baseChannels  = 32;                 % 64 = ~3.5x piu' lento (vedi SecPerIter)
numNoiseSteps = 500;
% Schedule coseno (Nichol & Dhariwal): usa meglio i passi rispetto al lineare,
% che a 64x64 porta a rumore quasi puro gia' per t > ~250.
cosS = 0.008;
tt_ = (0:numNoiseSteps)/numNoiseSteps;
f_ = cos((tt_ + cosS)/(1 + cosS) * pi/2).^2;
abar_ = f_/f_(1);
varianceSchedule = min(1 - abar_(2:end)./abar_(1:end-1), 0.999);

miniBatchSize = 16;
numIterations = 20000;
learnRate     = 2e-4;
pUncond       = 0.1;                % prob. di azzerare la classe (classifier-free guidance)
emaDecay      = 0.999;
evalEvery     = 500;                % loss su test (casi mai visti)
sampleEvery   = 5000;               % campionamento di controllo (lento su CPU)
checkpointEvery = 1000;
guidanceScale = 1;
resume        = false;

%% Dati (uint8 in memoria, conversione per batch)
D = load(dataFile);
XTrain = D.XTrain; YTrain = D.YTrain; XTest = D.XTest; YTest = D.YTest;
classNames = D.classNames;
assert(numel(classNames) == numClasses, "numClasses diverso dal numero di classi nei dati.");
N = size(XTrain, 4);
fprintf("Train: %d patch | Test: %d patch\n", N, size(XTest,4));
assert(size(XTrain,1) == imageSize, "imageSize diverso dalla risoluzione dei patch salvati.");

%% Sanity check del processo forward
alphaBar = cumprod(1 - varianceSchedule);
fprintf("alphaBar(T) = %.2e (deve essere ~0)\n", alphaBar(end));
x0 = toModelRange(XTrain(:,:,:,1));
figure; tiledlayout("flow");
nexttile; imshow((x0+1)/2); title("t = 0");
for tShow = [50 150 300 500]
    nexttile
    xt = applyNoiseToImage(x0, randn(size(x0),"single"), tShow, varianceSchedule);
    imshow(toImageRange(extractdata(xt))); title("t = " + tShow);
end
drawnow

%% Rete
if resume && isfile(ckptFile)
    C = load(ckptFile);
    net = C.net; netEMA = C.netEMA; iteration = C.iteration;
else
    net = createDiffusionNetwork(imageSize, numClasses, baseChannels);
    netEMA = net; iteration = 0;
end
fprintf("Parametri apprendibili: %.2f M\n", sum(cellfun(@numel, net.Learnables.Value))/1e6);

%% Training
avgG = []; avgSqG = [];
monitor = trainingProgressMonitor(Metrics=["Loss","TestLoss"], ...
    Info=["Iteration","SecPerIter"], XLabel="Iteration");
perm = randperm(N); pos = 1;
tStart = tic;

while iteration < numIterations && ~monitor.Stop
    iteration = iteration + 1;

    if pos + miniBatchSize - 1 > N
        perm = randperm(N); pos = 1;
    end
    idx = perm(pos:pos+miniBatchSize-1); pos = pos + miniBatchSize;

    batch  = augmentBatch(toModelRange(XTrain(:,:,:,idx)));
    labels = double(YTrain(idx))';
    labels(rand(1, miniBatchSize) < pUncond) = numClasses + 1;   % etichetta nulla

    X0    = dlarray(batch, "SSCB");
    noise = dlarray(randn(size(batch), "single"), "SSCB");
    t     = randi(numNoiseSteps, [1 miniBatchSize]);
    Xt    = applyNoiseToImage(X0, noise, t, varianceSchedule);

    tDL = dlarray(single(t), "CB");
    cDL = dlarray(single(labels), "CB");

    [loss, grads] = dlfeval(@modelLoss, net, Xt, tDL, cDL, noise);
    [net, avgG, avgSqG] = adamupdate(net, grads, avgG, avgSqG, iteration, ...
        learnRate, 0.9, 0.999);

    % EMA dei pesi (usata per il campionamento)
    d = min(emaDecay, (1 + iteration)/(10 + iteration));
    netEMA.Learnables.Value = cellfun(@(e,w) d*e + (1-d)*w, ...
        netEMA.Learnables.Value, net.Learnables.Value, UniformOutput=false);

    recordMetrics(monitor, iteration, Loss=double(extractdata(loss)));
    updateInfo(monitor, Iteration=iteration, SecPerIter=round(toc(tStart)/(iteration),2));
    monitor.Progress = 100*iteration/numIterations;

    if mod(iteration, evalEvery) == 0
        recordMetrics(monitor, iteration, TestLoss=evaluateTestLoss(net, XTest, YTest, ...
            numClasses, varianceSchedule, 64));
    end
    if mod(iteration, sampleEvery) == 0
        showSamples(netEMA, varianceSchedule, imageSize, numClasses, classNames, 3, guidanceScale);
    end
    if mod(iteration, checkpointEvery) == 0
        saveCheckpoint(ckptFile, net, netEMA, varianceSchedule, iteration, classNames, imageSize, numClasses);
    end
end
saveCheckpoint(ckptFile, net, netEMA, varianceSchedule, iteration, classNames, imageSize, numClasses);

%% Generazione finale: 8 immagini per classe
showSamples(netEMA, varianceSchedule, imageSize, numClasses, classNames, 8, guidanceScale);

%% ------------------------- funzioni locali -------------------------
function batch = augmentBatch(batch)
% Augmentation spaziale: gruppo diedrale (4 rotazioni x flip). Valida per
% l'istologia, che non ha orientamento canonico. Nessuna augmentation di
% colore: la colorazione e' gia' normalizzata (Macenko).
for i = 1:size(batch, 4)
    p = rot90(batch(:,:,:,i), randi(4) - 1);
    if rand < 0.5, p = flip(p, 2); end
    batch(:,:,:,i) = p;
end
end

function l = evaluateTestLoss(net, XTest, YTest, numClasses, varianceSchedule, n)
% MSE sul rumore su patch di casi MAI visti in training (spia di overfitting).
n = min(n, size(XTest, 4));
idx = randperm(size(XTest, 4), n);
X = toModelRange(XTest(:,:,:,idx));
noise = randn(size(X), "single");
t = randi(numel(varianceSchedule), [1 n]);
Xt = applyNoiseToImage(X, noise, t, varianceSchedule);
pred = predict(net, Xt, dlarray(single(t), "CB"), ...
    dlarray(single(double(YTest(idx))'), "CB"));
l = double(extractdata(mean((pred - dlarray(noise, "SSCB")).^2, "all")));
end

function saveCheckpoint(file, net, netEMA, varianceSchedule, iteration, classNames, imageSize, numClasses) %#ok<INUSD>
% Salva tutto il necessario per riprendere il training o generare immagini.
save(file, "net", "netEMA", "varianceSchedule", "iteration", "classNames", ...
    "imageSize", "numClasses", "-v7.3");
end

function showSamples(net, varianceSchedule, imageSize, numClasses, classNames, nPer, w)
classIdx = repelem(1:numClasses, nPer);
imgs = generateConditionalImages(net, varianceSchedule, imageSize, classIdx, numClasses, w);
figure; tiledlayout(numClasses, nPer, TileSpacing="none", Padding="compact");
for k = 1:numel(classIdx)
    nexttile; imshow(imgs(:,:,:,k));
    if mod(k-1, nPer) == 0, ylabel(classNames(classIdx(k))); end
end
drawnow
end