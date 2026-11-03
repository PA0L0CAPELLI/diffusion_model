function X = toImageRange(X)
%TOIMAGERANGE Riporta immagini dall'intervallo della rete [-1,1] a [0,1] per
%   imshow/imwrite, limitando i valori fuori range.
X = (min(max(X, -1), 1) + 1) / 2;
end
