function net = createDiffusionNetwork(numInputChannels)
%CREATEDIFFUSIONNETWORK Build a DDPM U-Net sized for 32x32 MNIST images.
%   net = createDiffusionNetwork(numInputChannels) returns an
%   initialized dlnetwork with two inputs:
%       "imageinput" - 32x32xnumInputChannels noisy image
%       "input"      - scalar noise step, format "CB"
%   and one output: the predicted noise, 32x32xnumInputChannels.
%
%   Architecture (mirrors the description in the MathWorks "Generate
%   Images Using Diffusion" example): an encoder-decoder U-Net built
%   from residual blocks, with self-attention blocks at the 16x16 and
%   8x8 resolutions. The noise step is encoded with a sinusoidal
%   position embedding and injected additively into every residual
%   block.
%
%   This is an independent reimplementation (MathWorks does not publish
%   the source of its own createDiffusionNetwork supporting file inline
%   in the doc page) — sized down a little for MNIST's simplicity, but
%   structurally equivalent to what the example describes.

if nargin < 1
    numInputChannels = 1;
end

embedDim = 128;

lgraph = layerGraph();

% ---- Image input ----
lgraph = addLayers(lgraph, imageInputLayer([32 32 numInputChannels], ...
    Normalization="none", Name="imageinput"));

% ---- Noise-step input -> sinusoidal embedding -> small MLP ----
lgraph = addLayers(lgraph, featureInputLayer(1, Normalization="none", Name="input"));
lgraph = addLayers(lgraph, [
    PositionEmbeddingLayer(embedDim, "posenc")
    fullyConnectedLayer(embedDim, Name="temb_fc1")
    swishLayer(Name="temb_swish")
    fullyConnectedLayer(embedDim, Name="temb_fc2")
    ]);
lgraph = connectLayers(lgraph, "input", "posenc");
tembSource = "temb_fc2";

% ---- Encoder ----
[lgraph, e0] = iResidualBlock(lgraph, "imageinput", "enc0",  64, tembSource);
[lgraph, e0] = iResidualBlock(lgraph, e0,           "enc0b", 64, tembSource);

[lgraph, d1] = iDownsample(lgraph, e0, "down1", 128);
[lgraph, e1] = iResidualBlock(lgraph, d1, "enc1",  128, tembSource);
[lgraph, e1] = iResidualBlock(lgraph, e1, "enc1b", 128, tembSource);
[lgraph, e1] = iAttentionBlock(lgraph, e1, "attn1", 128, 16, 16);

[lgraph, d2] = iDownsample(lgraph, e1, "down2", 256);
[lgraph, e2] = iResidualBlock(lgraph, d2, "enc2",  256, tembSource);
[lgraph, e2] = iResidualBlock(lgraph, e2, "enc2b", 256, tembSource);
[lgraph, e2] = iAttentionBlock(lgraph, e2, "attn2", 256, 8, 8);

% ---- Bottleneck (8x8) ----
[lgraph, b] = iResidualBlock(lgraph, e2, "bot0", 256, tembSource);
[lgraph, b] = iAttentionBlock(lgraph, b, "attnBot", 256, 8, 8);
[lgraph, b] = iResidualBlock(lgraph, b, "bot1", 256, tembSource);

% ---- Decoder ----
% up2: bottleneck is 8x8 -> upsample to 16x16, concat with encoder skip
% e1 (which is 16x16, 128 channels — NOT e2, which is also 8x8).
[lgraph, u2] = iUpsample(lgraph, b, "up2", 128);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, Name="cat2"));
lgraph = connectLayers(lgraph, u2, "cat2/in1");
lgraph = connectLayers(lgraph, e1, "cat2/in2");
[lgraph, g2] = iResidualBlock(lgraph, "cat2", "dec2", 128, tembSource);
[lgraph, g2] = iAttentionBlock(lgraph, g2, "dattn2", 128, 16, 16);

% up1: 16x16 -> upsample to 32x32, concat with encoder skip e0
% (32x32, 64 channels). This is the last upsample — the encoder only
% downsamples twice (32->16->8), so the decoder only needs to upsample
% twice to get back to 32x32.
[lgraph, u1] = iUpsample(lgraph, g2, "up1", 64);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, Name="cat1"));
lgraph = connectLayers(lgraph, u1, "cat1/in1");
lgraph = connectLayers(lgraph, e0, "cat1/in2");
[lgraph, g1] = iResidualBlock(lgraph, "cat1", "dec1", 64, tembSource);

% ---- Output projection back to numInputChannels ----
lgraph = addLayers(lgraph, convolution2dLayer(3, numInputChannels, Padding="same", Name="conv_out"));
lgraph = connectLayers(lgraph, g1, "conv_out");

net = dlnetwork(lgraph);

end

% ------------------------------------------------------------------
function [lgraph, outName] = iResidualBlock(lgraph, inName, blockName, numChannels, tembSource)
% Conv-BN-Swish -> add time embedding -> Conv-BN -> add skip -> Swish

c1 = blockName + "_conv1"; bn1 = blockName + "_bn1"; sw1 = blockName + "_swish1";
projFC = blockName + "_temb_fc";
tembAdd = blockName + "_tembadd";
c2 = blockName + "_conv2"; bn2 = blockName + "_bn2";
skipConv = blockName + "_skipconv";
addName = blockName + "_add";
sw2 = blockName + "_swish2";

lgraph = addLayers(lgraph, [
    convolution2dLayer(3, numChannels, Padding="same", Name=c1)
    batchNormalizationLayer(Name=bn1)
    swishLayer(Name=sw1)
    ]);
lgraph = connectLayers(lgraph, inName, c1);

lgraph = addLayers(lgraph, fullyConnectedLayer(numChannels, Name=projFC));
lgraph = connectLayers(lgraph, tembSource, projFC);

lgraph = addLayers(lgraph, BroadcastAdditionLayer(tembAdd));
lgraph = connectLayers(lgraph, sw1, tembAdd + "/in1");
lgraph = connectLayers(lgraph, projFC, tembAdd + "/in2");

lgraph = addLayers(lgraph, [
    convolution2dLayer(3, numChannels, Padding="same", Name=c2)
    batchNormalizationLayer(Name=bn2)
    ]);
lgraph = connectLayers(lgraph, tembAdd, c2);

% 1x1 conv on the skip path so channel counts always match, even when
% this block changes the number of channels.
lgraph = addLayers(lgraph, convolution2dLayer(1, numChannels, Padding="same", Name=skipConv));
lgraph = connectLayers(lgraph, inName, skipConv);

lgraph = addLayers(lgraph, additionLayer(2, Name=addName));
lgraph = connectLayers(lgraph, bn2, addName + "/in1");
lgraph = connectLayers(lgraph, skipConv, addName + "/in2");

lgraph = addLayers(lgraph, swishLayer(Name=sw2));
lgraph = connectLayers(lgraph, addName, sw2);

outName = sw2;
end

% ------------------------------------------------------------------
function [lgraph, outName] = iDownsample(lgraph, inName, blockName, numChannels)
name = blockName + "_conv";
lgraph = addLayers(lgraph, convolution2dLayer(3, numChannels, Stride=2, Padding="same", Name=name));
lgraph = connectLayers(lgraph, inName, name);
outName = name;
end

% ------------------------------------------------------------------
function [lgraph, outName] = iUpsample(lgraph, inName, blockName, numChannels)
% Nearest-neighbor resize followed by a normal convolution. This avoids
% the checkerboard artifacts that transposedConv2dLayer tends to bake
% into every output when its kernel size isn't a clean multiple of the
% stride (see https://distill.pub/2016/deconv-checkerboard/).
resizeName = blockName + "_resize";
convName = blockName + "_conv";

lgraph = addLayers(lgraph, resize2dLayer(Scale=[2 2], Method="nearest", Name=resizeName));
lgraph = connectLayers(lgraph, inName, resizeName);

lgraph = addLayers(lgraph, convolution2dLayer(3, numChannels, Padding="same", Name=convName));
lgraph = connectLayers(lgraph, resizeName, convName);

outName = convName;
end

% ------------------------------------------------------------------
function [lgraph, outName] = iAttentionBlock(lgraph, inName, blockName, numChannels, height, width)
flat = blockName + "_flatten";
attn = blockName + "_selfattn";
unflat = blockName + "_unflatten";
addName = blockName + "_add";
numHeads = 4;

lgraph = addLayers(lgraph, SpatialFlattenLayer(flat));
lgraph = connectLayers(lgraph, inName, flat);

lgraph = addLayers(lgraph, selfAttentionLayer(numHeads, numChannels, Name=attn));
lgraph = connectLayers(lgraph, flat, attn);

lgraph = addLayers(lgraph, SpatialUnflattenLayer(unflat, height, width));
lgraph = connectLayers(lgraph, attn, unflat);

lgraph = addLayers(lgraph, additionLayer(2, Name=addName));
lgraph = connectLayers(lgraph, inName, addName + "/in1");
lgraph = connectLayers(lgraph, unflat, addName + "/in2");

outName = addName;
end