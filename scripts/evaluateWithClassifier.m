%% Valutazione dei campioni condizionati con un classificatore addestrato su patch REALI
% 1) addestra (o carica) una piccola CNN su patch reali di train e la testa su casi MAI visti;
% 2) classifica le immagini sintetiche e confronta la classe richiesta con quella predetta;
% 3) confronta i colori medi di patch reali e sintetiche.
rng(0)
P = projectPaths();
D = load(P.patches, "XTrain", "YTrain", "XTest", "YTest", "classNames");
C = load(P.checkpoint);
classNames = D.classNames;
S = size(D.XTrain, 1);

%% 1) Classificatore su patch reali
if isfile(P.classifier)
    load(P.classifier, "clf");
else
    layers = [
        imageInputLayer([S S 3], Normalization="none")
        convolution2dLayer(3, 32, Padding="same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, Stride=2)
        convolution2dLayer(3, 64, Padding="same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, Stride=2)
        convolution2dLayer(3, 128, Padding="same")
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, Stride=2)
        convolution2dLayer(3, 128, Padding="same")
        batchNormalizationLayer
        reluLayer
        globalAveragePooling2dLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(numel(classNames))
        softmaxLayer
        classificationLayer];
    Xtr = single(D.XTrain) / 255;
    ytr = categorical(D.YTrain, 1:numel(classNames), classNames);
    aug = imageDataAugmenter(RandXReflection=true, RandYReflection=true);
    dsTr = augmentedImageDatastore([S S], Xtr, ytr, DataAugmentation=aug);
    opts = trainingOptions("adam", InitialLearnRate=1e-3, MaxEpochs=6, MiniBatchSize=128, ...
        Shuffle="every-epoch", Verbose=true, VerboseFrequency=50, Plots="none", ExecutionEnvironment="cpu");
    clf = trainNetwork(dsTr, layers, opts);
    save(P.classifier, "clf");
    clear Xtr
end

Xte = single(D.XTest) / 255;
yte = categorical(D.YTest, 1:numel(classNames), classNames);
pte = classify(clf, Xte);
accReal = mean(pte == yte);
fprintf("Accuratezza su patch REALI di test (casi mai visti): %.1f%% (caso = %.1f%%)\n", ...
    100*accReal, 100/numel(classNames));
figure("Name", "Classificatore - patch reali di test");
confusionchart(yte, pte, Title="Patch reali (test)");

%% 2) Campioni sintetici
numPerClass = 15;      % 45 immagini per valore di w (circa 25-40 min in totale su CPU)
cls = repelem(1:numel(classNames), numPerClass);
yreq = categorical(cls(:), 1:numel(classNames), classNames);
synth = struct();
wList = [1 2];
accSyn = zeros(numel(wList), 1);
for iw = 1:numel(wList)
    w = wList(iw);
    imgs = generateConditionalImages(C.netEMA, C.varianceSchedule, C.imageSize, cls, C.numClasses, w, true);
    p = classify(clf, imgs);
    accSyn(iw) = mean(p == yreq);
    fprintf("w=%g: classe richiesta = classe predetta nel %.1f%% dei campioni\n", w, 100*mean(p == yreq));
    figure("Name", "Sintetici w=" + w);
    confusionchart(yreq, p, Title="Sintetici, w=" + w + " (righe = classe richiesta)");
    synth.("w" + w) = imgs;
end

% Salva le accuratezze in results/classifier_metrics.csv
if ~isfolder(P.results), mkdir(P.results); end
metrics = table(["patch reali (test)"; "sintetici w=" + string(wList(:))], [accReal; accSyn], ...
    VariableNames=["Insieme", "Accuratezza"]);
writetable(metrics, fullfile(P.results, "classifier_metrics.csv"));
disp(metrics)

%% 3) Colori medi: reali vs sintetici
sub = single(D.XTrain(:,:,:, randperm(size(D.XTrain,4), 2000))) / 255;
mReal = squeeze(mean(sub, [1 2]));                        % 3 x n
mSyn  = squeeze(mean(synth.w2, [1 2]));
figure("Name", "Colore medio: reali vs sintetici (w=2)");
scatter(mReal(1,:) - mReal(3,:), mReal(2,:), 8, "filled", MarkerFaceAlpha=0.3); hold on
scatter(mSyn(1,:)  - mSyn(3,:),  mSyn(2,:),  20, "filled");
xlabel("R - B (negativo = piu' blu)"); ylabel("G medio (luminosita')");
legend("Patch reali (train)", "Sintetici w=2"); grid on