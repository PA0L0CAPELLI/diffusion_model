function P = projectPaths()
%PROJECTPATHS Percorsi assoluti delle cartelle e dei file del progetto.
%   P = projectPaths() funziona da qualunque cartella corrente: la radice
%   del progetto e' ricavata dalla posizione di questo file (functions/..).
%   Campi: root, data, checkpoints, generated, results, patches,
%   checkpoint, classifier.
root = string(fileparts(fileparts(mfilename("fullpath"))));

P.root        = root;
P.data        = fullfile(root, "data");
P.checkpoints = fullfile(root, "checkpoints");
P.generated   = fullfile(root, "generated");
P.results     = fullfile(root, "results");

P.patches     = fullfile(P.data, "LSCPatches.mat");                 % creato da prepareLSCDataset
P.checkpoint  = fullfile(P.checkpoints, "diffusionLSCCheckpoint.mat"); % creato da trainDiffusion
P.classifier  = fullfile(P.data, "lscClassifier.mat");              % creato da evaluateWithClassifier
end
