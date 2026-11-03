# Diffusion_model – Modello di diffusione condizionato su istologia LSC

Modello di diffusione DDPM condizionato per classe che genera patch istologiche
64×64 di tre linfomi (CLL, FL, MCL), implementato in MATLAB con un ciclo di
training personalizzato (Deep Learning Toolbox) e classifier-free guidance.

## Requisiti

- MATLAB con Deep Learning Toolbox, Image Processing Toolbox e
  Statistics and Machine Learning Toolbox (`prctile` in `macenkoNormalize`)
- Dataset LSC (Lymphoma Subtype Classification): una sottocartella per classe
  (`CLL/`, `FL/`, `MCL/`) dentro `data/LSC/`

## Struttura

| Cartella / file | Contenuto |
| --- | --- |
| `trainDiffusion.m` | script principale di training |
| `preprocessing/` | `prepareLSCDataset` (split per paziente, patch), `macenkoNormalize` |
| `layers/` | layer personalizzati: embedding di tempo e classe, somma broadcast, flatten/unflatten per l'attention |
| `functions/` | U-Net, processo forward, loss, campionamento, utility (`projectPaths`, `toModelRange`, `toImageRange`) |
| `scripts/` | generazione dal checkpoint, valutazione con classificatore, diagnostica |
| `data/` | dataset grezzo e `LSCPatches.mat` (non versionati) |
| `checkpoints/` | `diffusionLSCCheckpoint.mat` |
| `generated/` | PNG sintetici |
| `results/` | metriche in CSV prodotte dagli script |
| `docs/` | spiegazione di `trainDiffusion.m` |

## Esecuzione

Aprire `Diffusion_model.prj`. La prima volta, aggiungere le cartelle al path
del progetto (resta salvato nel progetto):

```matlab
proj = currentProject;
for f = ["functions" "layers" "preprocessing" "scripts"]
    addPath(proj, fullfile(proj.RootFolder, f));
end
```

Poi, in ordine:

```matlab
P = projectPaths();
prepareLSCDataset(fullfile(P.data, "LSC"))               % -> data/LSCPatches.mat (una volta)
trainDiffusion                                            % -> checkpoints/diffusionLSCCheckpoint.mat
generateFromCheckpoint                                    % -> generated/*.png, results/memorization.csv
evaluateWithClassifier                                    % -> results/classifier_metrics.csv
diagnoseSampling                                          % -> results/noise_mse.csv
```

Tutti i percorsi passano da `projectPaths()`, quindi gli script funzionano da
qualunque cartella corrente.
