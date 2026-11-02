function prepareLSCDataset(rootFolder, opts)
%PREPAREDATASET  Prepara il dataset LSC (CLL / FL / MCL) per la diffusione.
%
%   prepareLSCDataset("/path/LSC")
%   prepareLSCDataset("/path/LSC", PatchSize=128, OutputSize=64, ...)
%
% Passi:
%   1. legge le immagini (sottocartelle CLL, FL, MCL = etichette);
%   2. ricava l'ID del caso/paziente dal nome file;
%   3. split train/test STRATIFICATO PER CLASSE e a livello di caso
%      (nessun caso compare in entrambi gli insiemi);
%   4. normalizza la colorazione (Macenko) sull'immagine intera;
%   5. estrae patch quadrate a griglia (PatchSize px nativi, passo Stride),
%      scarta quelle senza tessuto e le ridimensiona a OutputSize;
%   6. salva tutto in LSCPatches.mat (uint8).
arguments
    rootFolder (1,1) string
    opts.PatchSize (1,1) double = 128        % lato del ritaglio in pixel nativi
    opts.Stride (1,1) double = 128           % passo della griglia (< PatchSize = overlap)
    opts.OutputSize (1,1) double = 64        % risoluzione di training
    opts.TrainFraction (1,1) double = 0.8
    opts.Seed (1,1) double = 42
    opts.StainNormalize (1,1) logical = true
    opts.MinTissueFraction (1,1) double = 0.5
    opts.OutFile (1,1) string = "LSCPatches.mat"
end

classNames = ["CLL", "FL", "MCL"];
imds = imageDatastore(rootFolder, IncludeSubfolders=true, LabelSource="foldernames", ...
    FileExtensions=[".jpg",".jpeg",".png",".tif",".tiff"]);
files = string(imds.Files);
labels = categorical(string(imds.Labels), classNames);
assert(~any(isundefined(labels)), "Trovate cartelle diverse da CLL/FL/MCL in %s", rootFolder);
classIdx = double(labels);                   % 1=CLL 2=FL 3=MCL

%% ID del caso (paziente)
caseIDs = extractCaseID(files);

%% Split per caso, stratificato per classe
rng(opts.Seed);
isTrain = false(numel(files), 1);
for k = 1:numel(classNames)
    cases = unique(caseIDs(classIdx == k));
    cases = cases(randperm(numel(cases)));
    nTrain = max(1, min(numel(cases)-1, round(opts.TrainFraction*numel(cases))));
    isTrain(classIdx == k & ismember(caseIDs, cases(1:nTrain))) = true;
    fprintf("%s: %d casi (%d train / %d test)\n", classNames(k), numel(cases), ...
        nTrain, numel(cases)-nTrain);
end
assert(isempty(intersect(caseIDs(isTrain), caseIDs(~isTrain))), ...
    "Data leakage: un caso compare sia in train che in test.");

%% Normalizzazione + patching
S = opts.OutputSize;
Xtr = {}; Ytr = {}; Ctr = {};
Xte = {}; Yte = {}; Cte = {};
for i = 1:numel(files)
    I = imread(files(i));
    if size(I,3) == 1, I = repmat(I, 1, 1, 3); end
    I = I(:,:,1:3);
    if opts.StainNormalize
        try
            I = macenkoNormalize(I);
        catch ME
            warning("Macenko fallito su %s (%s): uso l'immagine originale.", files(i), ME.message);
        end
    end
    P = iExtractPatches(I, opts.PatchSize, opts.Stride, opts.MinTissueFraction, S);
    n = size(P, 4);
    if n == 0, continue, end
    if isTrain(i)
        Xtr{end+1} = P; Ytr{end+1} = repmat(classIdx(i), n, 1); Ctr{end+1} = repmat(caseIDs(i), n, 1); %#ok<AGROW>
    else
        Xte{end+1} = P; Yte{end+1} = repmat(classIdx(i), n, 1); Cte{end+1} = repmat(caseIDs(i), n, 1); %#ok<AGROW>
    end
    if mod(i, 25) == 0, fprintf("  %d/%d immagini\n", i, numel(files)); end
end

XTrain = cat(4, Xtr{:}); YTrain = vertcat(Ytr{:}); caseTrain = vertcat(Ctr{:});
XTest  = cat(4, Xte{:}); YTest  = vertcat(Yte{:}); caseTest  = vertcat(Cte{:});

fprintf("Patch train: %d | test: %d\n", size(XTrain,4), size(XTest,4));
fprintf("Patch/classe (train): %s\n", mat2str(accumarray(YTrain, 1)'));
save(opts.OutFile, "XTrain","YTrain","caseTrain","XTest","YTest","caseTest", ...
    "classNames","opts","-v7.3");
end

% ------------------------------------------------------------------
function ids = extractCaseID(files)
% Nomi tipo "sj-05-894-R3_012" oppure "sj-05-768_011" (senza regione):
% caso = "sj-05-894" / "sj-05-768", R3 = regione/vetrino (opzionale),
% 012 = indice immagine. Se il pattern non corrisponde ogni immagine
% diventa un "caso" a se' (ATTENZIONE: possibile leakage tra immagini
% dello stesso paziente).
[~, names] = fileparts(files);
ids = strings(numel(files), 1);
nUnmatched = 0;
for i = 1:numel(files)
    tok = regexp(names(i), "^(.+?)(?:[-_]R\d+)?[-_]\d+$", "tokens", "once");
    if isempty(tok)
        ids(i) = names(i);
        nUnmatched = nUnmatched + 1;
    else
        ids(i) = string(tok(1));
    end
end
if nUnmatched > 0
    warning("%d/%d nomi file senza ID di caso riconoscibile: lo split NON e' garantito a livello di paziente. " + ...
        "Modifica extractCaseID o fornisci una tabella file->paziente.", nUnmatched, numel(files));
end
fprintf("Casi distinti trovati: %d\n", numel(unique(ids)));
end

function P = iExtractPatches(I, ps, stride, minTissue, S)
[h, w, ~] = size(I);
rows = 1:stride:(h - ps + 1);
cols = 1:stride:(w - ps + 1);
P = zeros(S, S, 3, numel(rows)*numel(cols), "uint8");
n = 0;
for r = rows
    for c = cols
        patch = I(r:r+ps-1, c:c+ps-1, :);
        g = rgb2gray(patch);
        if mean(g(:) < 215) < minTissue, continue, end   % troppo sfondo
        n = n + 1;
        P(:,:,:,n) = imresize(patch, [S S]);
    end
end
P = P(:,:,:,1:n);
end
