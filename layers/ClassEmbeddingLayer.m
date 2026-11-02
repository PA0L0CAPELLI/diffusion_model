classdef ClassEmbeddingLayer < nnet.layer.Layer & nnet.layer.Formattable
    % ClassEmbeddingLayer  Tabella di embedding apprendibile per le classi.
    % Input : 1 x B (formato "CB") con indici di classe 1..NumEmbeddings.
    %         L'ultimo indice (numClassi+1) e' l'etichetta "nulla" usata
    %         per il classifier-free guidance.
    % Output: EmbeddingDimension x B (formato "CB").

    properties
        EmbeddingDimension
        NumEmbeddings
    end
    properties (Learnable)
        Weights
    end

    methods
        function layer = ClassEmbeddingLayer(embeddingDimension, numEmbeddings, name)
            layer.EmbeddingDimension = embeddingDimension;
            layer.NumEmbeddings = numEmbeddings;
            layer.Name = name;
            layer.Description = "Class embedding (" + numEmbeddings + " x " + embeddingDimension + ")";
            layer.Weights = 0.02*randn(embeddingDimension, numEmbeddings, "single");
        end

        function Z = predict(layer, X)
            idx = double(extractdata(stripdims(X)));
            idx = reshape(idx, 1, []);
            idx = min(max(round(idx), 1), layer.NumEmbeddings);   % robusto ai dati fittizi di inizializzazione
            Z = dlarray(layer.Weights(:, idx), "CB");
        end
    end
end
