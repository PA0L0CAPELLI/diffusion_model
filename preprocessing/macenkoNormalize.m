function Inorm = macenkoNormalize(I, opts)
%MACENKONORMALIZE Normalizzazione della colorazione H&E (metodo di Macenko).
%   Inorm = macenkoNormalize(I) stima i vettori di colore Ematossilina/Eosina
%   dell'immagine RGB uint8 I e la riproietta sui vettori di riferimento
%   standard (valori di default delle implementazioni di riferimento).
%   Richiede Statistics and Machine Learning Toolbox (prctile).
%
%   Se la stima e' degenere (poco tessuto, matrice singolare) restituisce
%   l'immagine originale.
arguments
    I uint8
    opts.Io (1,1) double = 240
    opts.Alpha (1,1) double = 1
    opts.Beta (1,1) double = 0.15
    opts.HERef (3,2) double = [0.5626 0.2159; 0.7201 0.8012; 0.4062 0.5581]
    opts.MaxCRef (2,1) double = [1.9705; 1.0308]
end

[h, w, ~] = size(I);
OD = -log((double(reshape(I, [], 3)) + 1) ./ opts.Io);   % N x 3

keep = all(OD > opts.Beta, 2);                            % scarta sfondo
if nnz(keep) < 1000
    Inorm = I; return
end
ODhat = OD(keep, :);

[V, D] = eig(cov(ODhat));
[~, order] = sort(diag(D), "descend");
V = V(:, order(1:2));
if V(1,1) < 0, V(:,1) = -V(:,1); end
if V(1,2) < 0, V(:,2) = -V(:,2); end

That = ODhat * V;
phi = atan2(That(:,2), That(:,1));
minPhi = prctile(phi, opts.Alpha);
maxPhi = prctile(phi, 100 - opts.Alpha);
vMin = V * [cos(minPhi); sin(minPhi)];
vMax = V * [cos(maxPhi); sin(maxPhi)];

if vMin(1) > vMax(1)
    HE = [vMin, vMax];     % colonna 1 = Ematossilina, colonna 2 = Eosina
else
    HE = [vMax, vMin];
end

C = HE \ OD';                                % 2 x N concentrazioni
maxC = prctile(C', 99)';                     % 2 x 1
C = C .* (opts.MaxCRef ./ maxC);

Irec = opts.Io * exp(-opts.HERef * C);       % 3 x N
if any(~isfinite(Irec(:)))
    Inorm = I; return
end
Irec = min(max(Irec, 0), 255);
Inorm = uint8(reshape(Irec', h, w, 3));
end
