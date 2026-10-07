classdef SpatialUnflattenLayer < nnet.layer.Layer & nnet.layer.Formattable
    % SpatialUnflattenLayer  Reshape a CBT sequence (C x B x T) produced
    % by selfAttentionLayer back into an SSCB feature map
    % (Height x Width x C x B). Height and Width must be supplied at
    % construction time (they are known statically from the network
    % architecture, since T = Height*Width).

    properties
        Height
        Width
    end

    methods
        function layer = SpatialUnflattenLayer(name, height, width)
            layer.Name = name;
            layer.Height = height;
            layer.Width = width;
        end

        function Z = predict(layer, X)
            Xp = stripdims(X);                  % C x B x T
            [c, b, ~] = size(Xp);
            Xr = permute(Xp, [3 1 2]);           % T x C x B
            Xr = reshape(Xr, layer.Height, layer.Width, c, b);
            Z = dlarray(Xr, "SSCB");
        end
    end
end
