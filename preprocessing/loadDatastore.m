function imds = loadDatastore(source, folder)
%LOADMNISTDATASTORE Get an imageDatastore of MNIST-style digit images.
%
%   imds = loadMNISTDatastore("idx", "/path/to/mnist/folder")
%       Reads the classic Yann LeCun IDX files:
%       train-images-idx3-ubyte(.gz), train-labels-idx1-ubyte(.gz)
%       from the given folder, writes them out as PNGs into a temp
%       folder (once), and returns an imageDatastore over that folder.
%
%   imds = loadMNISTDatastore("folders", "/path/to/mnist_png")
%       Assumes the folder already contains one subfolder per digit
%       (0..9), each full of PNG/JPG images — e.g. the layout you get
%       from most "MNIST as PNG" Kaggle/GitHub mirrors.
%
%   imds = loadMNISTDatastore("builtin")
%       No download needed. Uses the small synthetic digit dataset
%       shipped with Deep Learning Toolbox
%       (matlabroot/toolbox/nnet/nndemos/nndatasets/DigitDataset,
%       10,000 28x28 grayscale digit images). This is NOT the original
%       MNIST data, but it is drop-in compatible with this pipeline and
%       is useful for getting the training loop running before you
%       plug in real MNIST.
%
%   In all cases the returned imds has IncludeSubfolders=true and
%   LabelSource="foldernames", matching the pattern used in the
%   MathWorks "Generate Images Using Diffusion" example.

if nargin < 1
    source = "builtin";
end

switch lower(string(source))
    case "builtin"
        digitDatasetPath = fullfile(matlabroot, "toolbox", "nnet", ...
            "nndemos", "nndatasets", "DigitDataset");
        imds = imageDatastore(digitDatasetPath, ...
            IncludeSubfolders=true, LabelSource="foldernames");

    case "idx"
        imds = iLoadFromIDX(folder);

    case "folders"
        imds = iLoadFromFolders(folder);

    otherwise
        error("loadMNISTDatastore:BadUsage", ...
            "Call as loadMNISTDatastore(""idx"",folder), loadMNISTDatastore(""folders"",folder), or loadMNISTDatastore(""builtin"").");
end
end

% ------------------------------------------------------------------
function imds = iLoadFromFolders(folder)
imds = imageDatastore(folder, IncludeSubfolders=true, LabelSource="foldernames");
end

% ------------------------------------------------------------------
function imds = iLoadFromIDX(folder)
% Reads the four classic MNIST IDX files (train split) and rasterizes
% them to PNG files under one folder per digit, then returns an
% imageDatastore over that folder. Only needs to run once; subsequent
% calls reuse the cached PNGs.

outFolder = fullfile(tempdir, "MNISTPngCache");
if isfolder(outFolder) && ~isempty(dir(fullfile(outFolder, "**", "*.png")))
    imds = imageDatastore(outFolder, IncludeSubfolders=true, LabelSource="foldernames");
    return
end

imagesFile = iFindFile(folder, "train-images-idx3-ubyte");
labelsFile = iFindFile(folder, "train-labels-idx1-ubyte");

images = iReadIDXImages(imagesFile);   % H x W x N uint8
labels = iReadIDXLabels(labelsFile);   % N x 1

for d = 0:9
    mkdir(fullfile(outFolder, string(d)));
end

fprintf("Writing MNIST PNGs to %s (one-time cache)...\n", outFolder);
for i = 1:size(images,3)
    lbl = labels(i);
    fname = fullfile(outFolder, string(lbl), sprintf("%06d.png", i));
    imwrite(images(:,:,i), fname);
end

imds = imageDatastore(outFolder, IncludeSubfolders=true, LabelSource="foldernames");
end

% ------------------------------------------------------------------
function f = iFindFile(folder, baseName)
candidates = [fullfile(folder, baseName), fullfile(folder, baseName + ".gz")];
f = "";
for c = candidates
    if isfile(c)
        f = c;
        break
    end
end
if f == ""
    error("loadMNISTDatastore:FileNotFound", ...
        "Could not find %s or %s.gz in %s", baseName, baseName, folder);
end
if endsWith(f, ".gz")
    gunzip(f);
    f = extractBefore(f, ".gz");
end
end

% ------------------------------------------------------------------
function images = iReadIDXImages(filename)
fid = fopen(filename, "r", "b");
magic = fread(fid, 1, "int32"); %#ok<NASGU>
numImages = fread(fid, 1, "int32");
numRows = fread(fid, 1, "int32");
numCols = fread(fid, 1, "int32");
images = fread(fid, numRows*numCols*numImages, "uint8=>uint8");
fclose(fid);
images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]);
end

% ------------------------------------------------------------------
function labels = iReadIDXLabels(filename)
fid = fopen(filename, "r", "b");
magic = fread(fid, 1, "int32"); %#ok<NASGU>
numLabels = fread(fid, 1, "int32");
labels = fread(fid, numLabels, "uint8=>double");
fclose(fid);
end
