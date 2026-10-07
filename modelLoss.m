function [loss, gradients, state] = modelLoss(net, X, Y, T)
%MODELLOSS Mean squared error between predicted and actual added noise.
%   X: mini-batch of noisy images. Y: noise step for each image.
%   T: the actual (target) noise that was added to each image.
    [noisePrediction, state] = forward(net, X, Y);          % Capture state here
    loss = mse(noisePrediction, T);
    gradients = dlgradient(loss, net.Learnables);
end