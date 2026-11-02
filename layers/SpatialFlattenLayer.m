classdef SpatialFlattenLayer < nnet.layer.Layer & nnet.layer.Formattable
    % SpatialFlattenLayer  Reshape an SSCB feature map (H x W x C x B)
    % into a CBT sequence (C x B x T, T = H*W) so it can be passed to
    % selfAttentionLayer.

    methods
        function layer = SpatialFlattenLayer(name)
            layer.Name = name;
        end

        function Z = predict(~, X)
            Xp = stripdims(X);                 % H x W x C x B
            [h, w, c, b] = size(Xp);
            Xr = reshape(Xp, h*w, c, b);        % T x C x B
            Xr = permute(Xr, [2 3 1]);          % C x B x T
            Z = dlarray(Xr, "CBT");
        end
    end
end
