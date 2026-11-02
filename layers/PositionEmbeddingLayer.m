classdef PositionEmbeddingLayer < nnet.layer.Layer & nnet.layer.Formattable
    % PositionEmbeddingLayer  Embedding sinusoidale del passo di rumore t.
    % Input : 1 x B, formato "CB".   Output: EmbeddingDimension x B, "CB".

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

            t = reshape(stripdims(X), 1, []);                 % 1 x B
            expTerm = -log(10000) * (0:halfDim-1)' / max(halfDim-1, 1);
            freqs = exp(expTerm);                             % halfDim x 1
            args = freqs * t;                                 % halfDim x B
            emb = [sin(args); cos(args)];

            if size(emb,1) < d
                emb = [emb; zeros(d - size(emb,1), size(emb,2), "like", emb)];
            end
            Z = dlarray(emb, "CB");        % <-- formato esplicito (layer Formattable)
        end
    end
end
