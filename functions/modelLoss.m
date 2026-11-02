function [loss, gradients] = modelLoss(net, Xt, t, c, targetNoise)
%MODELLOSS MSE tra rumore predetto e rumore reale, condizionato su (t, classe).
noisePrediction = forward(net, Xt, t, c);
% MSE medio per elemento (calcolo esplicito: indipendente dalla release di MATLAB)
loss = mean((noisePrediction - targetNoise).^2, "all");
gradients = dlgradient(loss, net.Learnables);
end