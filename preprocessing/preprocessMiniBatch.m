function X = preprocessMiniBatch(data)
%PREPROCESSMINIBATCH Concatenate a mini-batch and rescale to [-1, 1].

X = cat(4, data{:});
X = rescale(X, -1, 1, InputMin=0, InputMax=255);
end