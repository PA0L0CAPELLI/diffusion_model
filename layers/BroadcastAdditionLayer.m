classdef BroadcastAdditionLayer < nnet.layer.Layer & nnet.layer.Formattable
    % BroadcastAdditionLayer  Add a per-channel vector to every spatial
    % location of a feature map. Used to inject the time-step embedding
    % into each residual block.
    %
    % Inputs:
    %   in1: SSCB feature map, size H x W x C x B
    %   in2: CB   embedding,   size C x B
    % Output:
    %   SSCB feature map, size H x W x C x B

    methods
        function layer = BroadcastAdditionLayer(name)
            layer.Name = name;
            layer.NumInputs = 2;
            layer.InputNames = {'in1','in2'};
            layer.NumOutputs = 1;
        end

        function Z = predict(~, X, E)
            Ep = stripdims(E);                       % C x B
            [c, b] = size(Ep);
            Er = reshape(Ep, 1, 1, c, b);
            Efmt = dlarray(Er, "SSCB");
            Z = X + Efmt;                             % relies on dlarray broadcasting
        end
    end
end
