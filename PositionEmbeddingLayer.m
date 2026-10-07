classdef PositionEmbeddingLayer < nnet.layer.Layer & nnet.layer.Formattable
    % PositionEmbeddingLayer  Sinusoidal embedding of a scalar noise step.
    %
    % Input:  1(C) x N(B) dlarray, format "CB" — the noise step t for
    %         each observation in the mini-batch.
    % Output: EmbeddingDimension(C) x N(B) dlarray, format "CB".

    properties
        EmbeddingDimension
    end

    methods
        function layer = PositionEmbeddingLayer(embeddingDimension, name)
            layer.EmbeddingDimension = embeddingDimension;
            layer.Name = name;
            layer.Description = "Sinusoidal position embedding (dim=" + embeddingDimension + ")";
        end

        function Z = predict(layer, X)
            d = layer.EmbeddingDimension;
            halfDim = floor(d/2);

            Xp = stripdims(X);
            t = reshape(Xp, 1, []);                 % 1 x N

            expTerm = -log(10000) * (0:halfDim-1)' / max(halfDim-1,1);
            freqs = exp(expTerm);                   % halfDim x 1

            args = freqs * t;                       % halfDim x N
            emb = [sin(args); cos(args)];            % (2*halfDim) x N

            if size(emb,1) < d
                pad = zeros(d - size(emb,1), size(emb,2), "like", emb);
                emb = [emb; pad];
            end

            Z = dlarray(emb);
        end
    end
end
