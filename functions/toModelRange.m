function X = toModelRange(X)
%TOMODELRANGE Converte immagini uint8 [0,255] nell'intervallo [-1,1] usato dalla rete.
X = single(X) / 127.5 - 1;
end
