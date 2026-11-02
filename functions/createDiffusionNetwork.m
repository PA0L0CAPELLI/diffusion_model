function net = createDiffusionNetwork(imageSize, numClasses, baseChannels)
%CREATEDIFFUSIONNETWORK U-Net condizionata (tempo + classe) per la diffusione.
%   Ingressi (in quest'ordine): immagine [S S 3 B], passo t [1 B], classe [1 B].
%   Classe numClasses+1 = "nessuna classe" (classifier-free guidance).
%   imageSize deve essere multiplo di 8. Per 64x64: risoluzioni 64/32/16/8,
%   attention solo a 16x16 e 8x8 (leggera su CPU).
arguments
    imageSize (1,1) double = 64
    numClasses (1,1) double = 3
    baseChannels (1,1) double = 64
end
assert(mod(imageSize, 8) == 0, "imageSize deve essere multiplo di 8.");

numInputChannels = 3;
embedDim = 128;
c1 = baseChannels; c2 = 2*baseChannels; c3 = 2*baseChannels; c4 = 4*baseChannels;
s0 = imageSize; s1 = s0/2; s2 = s1/2; s3 = s2/2; %#ok<NASGU>

lgraph = layerGraph();

% ---- Ingressi (l'ordine di aggiunta definisce l'ordine in predict/forward) ----
lgraph = addLayers(lgraph, imageInputLayer([imageSize imageSize numInputChannels], ...
    Normalization="none", Name="imageinput"));
lgraph = addLayers(lgraph, featureInputLayer(1, Normalization="none", Name="input"));
lgraph = addLayers(lgraph, featureInputLayer(1, Normalization="none", Name="classinput"));

% ---- Embedding del tempo ----
lgraph = addLayers(lgraph, [
    PositionEmbeddingLayer(embedDim, "posenc")
    fullyConnectedLayer(embedDim, Name="temb_fc1")
    swishLayer(Name="temb_swish")
    fullyConnectedLayer(embedDim, Name="temb_fc2")]);
lgraph = connectLayers(lgraph, "input", "posenc");

% ---- Embedding della classe, sommato a quello del tempo ----
lgraph = addLayers(lgraph, ClassEmbeddingLayer(embedDim, numClasses + 1, "classemb"));
lgraph = connectLayers(lgraph, "classinput", "classemb");
lgraph = addLayers(lgraph, additionLayer(2, Name="cond_add"));
lgraph = connectLayers(lgraph, "temb_fc2", "cond_add/in1");
lgraph = connectLayers(lgraph, "classemb", "cond_add/in2");
cond = "cond_add";

% ---- Encoder ----
[lgraph, e0] = iResidualBlock(lgraph, "imageinput", "enc0",  c1, cond);
[lgraph, e0] = iResidualBlock(lgraph, e0,           "enc0b", c1, cond);

[lgraph, d1] = iDownsample(lgraph, e0, "down1", c2);
[lgraph, e1] = iResidualBlock(lgraph, d1, "enc1",  c2, cond);
[lgraph, e1] = iResidualBlock(lgraph, e1, "enc1b", c2, cond);

[lgraph, d2] = iDownsample(lgraph, e1, "down2", c3);
[lgraph, e2] = iResidualBlock(lgraph, d2, "enc2",  c3, cond);
[lgraph, e2] = iResidualBlock(lgraph, e2, "enc2b", c3, cond);
[lgraph, e2] = iAttentionBlock(lgraph, e2, "attn1", c3, s2, s2);

% ---- Bottleneck ----
[lgraph, d3] = iDownsample(lgraph, e2, "down3", c4);
[lgraph, b]  = iResidualBlock(lgraph, d3, "bot0", c4, cond);
[lgraph, b]  = iAttentionBlock(lgraph, b, "attnBot", c4, s3, s3);
[lgraph, b]  = iResidualBlock(lgraph, b, "bot1", c4, cond);

% ---- Decoder ----
[lgraph, u3] = iUpsample(lgraph, b, "up3", c3);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, Name="cat3"));
lgraph = connectLayers(lgraph, u3, "cat3/in1");
lgraph = connectLayers(lgraph, e2, "cat3/in2");
[lgraph, g3] = iResidualBlock(lgraph, "cat3", "dec3", c3, cond);
[lgraph, g3] = iAttentionBlock(lgraph, g3, "dattn3", c3, s2, s2);

[lgraph, u2] = iUpsample(lgraph, g3, "up2", c2);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, Name="cat2"));
lgraph = connectLayers(lgraph, u2, "cat2/in1");
lgraph = connectLayers(lgraph, e1, "cat2/in2");
[lgraph, g2] = iResidualBlock(lgraph, "cat2", "dec2", c2, cond);

[lgraph, u1] = iUpsample(lgraph, g2, "up1", c1);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, Name="cat1"));
lgraph = connectLayers(lgraph, u1, "cat1/in1");
lgraph = connectLayers(lgraph, e0, "cat1/in2");
[lgraph, g1] = iResidualBlock(lgraph, "cat1", "dec1", c1, cond);

lgraph = addLayers(lgraph, convolution2dLayer(3, numInputChannels, Padding="same", Name="conv_out"));
lgraph = connectLayers(lgraph, g1, "conv_out");

net = dlnetwork(lgraph);
end

% ------------------------------------------------------------------
function [lgraph, outName] = iResidualBlock(lgraph, inName, blockName, numChannels, condSource)
% Conv-GN-Swish -> + embedding(t,classe) -> Conv-GN -> + skip -> Swish
% (GroupNorm al posto di BatchNorm: nessuna statistica di popolazione,
%  quindi nessun disallineamento tra training e campionamento).
c1 = blockName + "_conv1"; gn1 = blockName + "_gn1"; sw1 = blockName + "_swish1";
projFC = blockName + "_cond_fc";
tembAdd = blockName + "_tembadd";
c2 = blockName + "_conv2"; gn2 = blockName + "_gn2";
skipConv = blockName + "_skipconv";
addName = blockName + "_add";
sw2 = blockName + "_swish2";

lgraph = addLayers(lgraph, [
    convolution2dLayer(3, numChannels, Padding="same", Name=c1)
    groupNormalizationLayer(8, Name=gn1)
    swishLayer(Name=sw1)]);
lgraph = connectLayers(lgraph, inName, c1);

lgraph = addLayers(lgraph, fullyConnectedLayer(numChannels, Name=projFC));
lgraph = connectLayers(lgraph, condSource, projFC);

lgraph = addLayers(lgraph, BroadcastAdditionLayer(tembAdd));
lgraph = connectLayers(lgraph, sw1, tembAdd + "/in1");
lgraph = connectLayers(lgraph, projFC, tembAdd + "/in2");

lgraph = addLayers(lgraph, [
    convolution2dLayer(3, numChannels, Padding="same", Name=c2)
    groupNormalizationLayer(8, Name=gn2)]);
lgraph = connectLayers(lgraph, tembAdd, c2);

lgraph = addLayers(lgraph, convolution2dLayer(1, numChannels, Padding="same", Name=skipConv));
lgraph = connectLayers(lgraph, inName, skipConv);

lgraph = addLayers(lgraph, additionLayer(2, Name=addName));
lgraph = connectLayers(lgraph, gn2, addName + "/in1");
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
