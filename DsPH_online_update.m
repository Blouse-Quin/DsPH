function [codeImage, codeText, imageMap, textMap, middlePara, info] = DsPH_online_update(...
    imagePaired, textPaired, imageUnpaired, textUnpaired, labelsPaired, ...
    labelsUnpairedImage, labelsUnpairedText, params, middlePara)
% Update DsPH with one later batch using previous sufficient statistics.

G = DsPH_normalize_rows(labelsPaired);
GI = DsPH_normalize_rows(labelsUnpairedImage);
GT = DsPH_normalize_rows(labelsUnpairedText);
nbits = params.nbits;
nPaired = size(labelsPaired, 1);
nImageUnpaired = size(labelsUnpairedImage, 1);
nTextUnpaired = size(labelsUnpairedText, 1);

B = DsPH_sign(randn(nPaired, nbits));
BI = DsPH_sign(randn(nImageUnpaired, nbits));
BT = DsPH_sign(randn(nTextUnpaired, nbits));
V = randn(nPaired, nbits);
R = randn(nbits, nbits);
[U, ~, ~] = svd(R);
R = U(:, 1:nbits);
C = ones(size(labelsPaired));
C(labelsPaired == 0) = -1;
E = zeros(size(C));

for iteration = 1:params.maxIter
    systemMatrix = middlePara{1} - V' * V;
    rightHandSide = middlePara{2} + V' * (labelsPaired + C .* E);
    if rcond(systemMatrix) < 1e-12
        regularization = 1e-8 * max(norm(systemMatrix, 'fro'), 1);
        systemMatrix = systemMatrix + regularization * eye(nbits);
    end
    P = systemMatrix \ ...
        rightHandSide;
    Z = params.alpha2 * B * R' ...
        + params.alpha1 * (labelsPaired + C .* E) * P' ...
        + 2 * nbits * G * (middlePara{3} + G' * B) ...
        - nbits * ones(nPaired, 1) * (middlePara{4} + ones(1, nPaired) * B) ...
        + 2 * params.beta1 * nbits * G * (middlePara{12} + GI' * BI) ...
        - params.beta1 * nbits * ones(nPaired, 1) * (middlePara{13} + ones(1, nImageUnpaired) * BI) ...
        + 2 * params.beta2 * nbits * G * (middlePara{14} + GT' * BT) ...
        - params.beta2 * nbits * ones(nPaired, 1) * (middlePara{15} + ones(1, nTextUnpaired) * BT);
    V = DsPH_orthogonal_update(Z, nPaired, nbits);
    [UB, ~, UA] = svd(middlePara{5} + B' * V);
    R = UA * UB';
    B = DsPH_sign(2 * nbits * G * (middlePara{6} + G' * V) ...
        - nbits * ones(nPaired, 1) * (middlePara{7} + ones(1, nPaired) * V) ...
        + params.alpha2 * V * R);
    BI = DsPH_sign(2 * params.beta1 * nbits * GI * (middlePara{16} + G' * V) ...
        - params.beta1 * nbits * ones(nImageUnpaired, 1) * (middlePara{17} + ones(1, nPaired) * V) ...
        + 2 * params.beta3 * nbits * GI * (middlePara{20} + GT' * BT) ...
        - params.beta3 * nbits * ones(nImageUnpaired, 1) * (middlePara{21} + ones(1, nTextUnpaired) * BT));
    BT = DsPH_sign(2 * params.beta2 * nbits * GT * (middlePara{18} + G' * V) ...
        - params.beta2 * nbits * ones(nTextUnpaired, 1) * (middlePara{19} + ones(1, nPaired) * V) ...
        + 2 * params.beta3 * nbits * GT * (middlePara{22} + GI' * BI) ...
        - params.beta3 * nbits * ones(nTextUnpaired, 1) * (middlePara{23} + ones(1, nImageUnpaired) * BI));
    E = max((V * P - labelsPaired) .* E, 0);
end

currentX = [imagePaired; imageUnpaired];
currentY = [textPaired; textUnpaired];
middlePara{1} = middlePara{1} + V' * V;
middlePara{2} = middlePara{2} + V' * (labelsPaired + C .* E);
middlePara{3} = middlePara{3} + G' * B;
middlePara{4} = middlePara{4} + ones(1, nPaired) * B;
middlePara{5} = middlePara{5} + B' * V;
middlePara{6} = middlePara{6} + G' * V;
middlePara{7} = middlePara{7} + ones(1, nPaired) * V;
middlePara{8} = middlePara{8} + currentX' * currentX;
middlePara{9} = middlePara{9} + currentY' * currentY;
middlePara{10} = middlePara{10} + currentX' * [B; BI];
middlePara{11} = middlePara{11} + currentY' * [B; BT];
middlePara{12} = middlePara{12} + GI' * BI;
middlePara{13} = middlePara{13} + ones(1, nImageUnpaired) * BI;
middlePara{14} = middlePara{14} + GT' * BT;
middlePara{15} = middlePara{15} + ones(1, nTextUnpaired) * BT;
middlePara{16} = middlePara{16} + G' * V;
middlePara{17} = middlePara{17} + ones(1, nPaired) * V;
middlePara{18} = middlePara{18} + G' * V;
middlePara{19} = middlePara{19} + ones(1, nPaired) * V;
middlePara{20} = middlePara{20} + GT' * BT;
middlePara{21} = middlePara{21} + ones(1, nTextUnpaired) * BT;
middlePara{22} = middlePara{22} + GI' * BI;
middlePara{23} = middlePara{23} + ones(1, nImageUnpaired) * BI;

imageMap = (middlePara{8} + params.xi * eye(size(middlePara{8}))) \ middlePara{10};
textMap = (middlePara{9} + params.xi * eye(size(middlePara{9}))) \ middlePara{11};
codeImage = [B; BI];
codeText = [B; BT];
info.iterations = iteration;
end

function V = DsPH_orthogonal_update(Z, nSamples, nbits)
centered = Z - (1 / nSamples) * ones(nSamples, 1) * (ones(1, nSamples) * Z);
[~, singularValues, rightVectors] = svd(centered' * centered);
valid = diag(singularValues) > 1e-6;
Q = rightVectors(:, valid);
    QNull = orth(rightVectors(:, ~valid));
if any(valid)
    P = centered * (Q / sqrt(singularValues(valid, valid)));
else
    P = zeros(nSamples, 0);
end
missing = nbits - size(P, 2);
if missing > 0
    PNull = orth(randn(nSamples, missing));
    QNull = QNull(:, 1:missing);
else
    PNull = zeros(nSamples, 0);
    QNull = zeros(size(Z, 2), 0);
end
V = sqrt(nSamples) * [P, PNull] * [Q, QNull]';
end