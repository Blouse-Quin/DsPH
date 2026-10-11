function [codeImage, codeText, imageMap, textMap, middlePara, info] = DsPH_start(...
    imagePaired, textPaired, imageUnpaired, textUnpaired, labelsPaired, ...
    labelsUnpairedImage, labelsUnpairedText, params)
% Train DsPH on the first partially paired batch.

DsPH_validate_batch(imagePaired, textPaired, imageUnpaired, textUnpaired, ...
    labelsPaired, labelsUnpairedImage, labelsUnpairedText, params);

X = [imagePaired; imageUnpaired];
Y = [textPaired; textUnpaired];
G = DsPH_normalize_rows(labelsPaired);
GI = DsPH_normalize_rows(labelsUnpairedImage);
GT = DsPH_normalize_rows(labelsUnpairedText);

maxIter = params.maxIter;
alpha1 = params.alpha1;
alpha2 = params.alpha2;
beta1 = params.beta1;
beta2 = params.beta2;
beta3 = params.beta3;
xi = params.xi;
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
previousLoss = 0;

for iteration = 1:maxIter
    P = (-(V' * V)) \ (V' * (labelsPaired + C .* E));

    Z = alpha2 * B * R' ...
        + alpha1 * (labelsPaired + C .* E) * P' ...
        + 2 * nbits * G * (G' * B) ...
        - nbits * ones(nPaired, 1) * (ones(1, nPaired) * B) ...
        + 2 * beta1 * nbits * G * (GI' * BI) ...
        - beta1 * nbits * ones(nPaired, 1) * (ones(1, nImageUnpaired) * BI) ...
        + 2 * beta2 * nbits * G * (GT' * BT) ...
        - beta2 * nbits * ones(nPaired, 1) * (ones(1, nTextUnpaired) * BT);

    V = DsPH_orthogonal_update(Z, nPaired, nbits);
    [UB, ~, UA] = svd(B' * V);
    R = UA * UB';

    B = DsPH_sign(2 * nbits * G * (G' * V) ...
        - nbits * ones(nPaired, 1) * (ones(1, nPaired) * V) ...
        + alpha2 * V * R);
    BI = DsPH_sign(2 * beta1 * nbits * GI * (G' * V) ...
        - beta1 * nbits * ones(nImageUnpaired, 1) * (ones(1, nPaired) * V) ...
        + 2 * beta3 * nbits * GI * (GT' * BT) ...
        - beta3 * nbits * ones(nImageUnpaired, 1) * (ones(1, nTextUnpaired) * BT));
    BT = DsPH_sign(2 * beta2 * nbits * GT * (G' * V) ...
        - beta2 * nbits * ones(nTextUnpaired, 1) * (ones(1, nPaired) * V) ...
        + 2 * beta3 * nbits * GT * (GI' * BI) ...
        - beta3 * nbits * ones(nTextUnpaired, 1) * (ones(1, nImageUnpaired) * BI));

    E = max((V * P - labelsPaired) .* E, 0);
    loss = norm(B * B' - nbits * (G * G')) ...
        + beta1 * norm(BI * B' - nbits * (GI * G')) ...
        + beta2 * norm(BT * B' - nbits * (GT * G')) ...
        + alpha1 * norm(labelsPaired + C .* E - B * P, 'fro');
    if params.verbose
        fprintf('DsPH iteration %d, loss %.6g\n', iteration, loss);
    end
    if iteration > 1 && abs(previousLoss - loss) < params.epsilon
        break;
    end
    previousLoss = loss;
end

middlePara = DsPH_build_statistics(X, Y, labelsPaired, G, GI, GT, B, BI, BT, V, R, P, C, E);
imageMap = (middlePara{8} + xi * eye(size(X, 2))) \ middlePara{10};
textMap = (middlePara{9} + xi * eye(size(Y, 2))) \ middlePara{11};
codeImage = [B; BI];
codeText = [B; BT];
info.iterations = iteration;
info.loss = loss;
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

function middlePara = DsPH_build_statistics(X, Y, labelsPaired, G, GI, GT, B, BI, BT, V, R, P, C, E)
nPaired = size(B, 1);
nImageUnpaired = size(BI, 1);
nTextUnpaired = size(BT, 1);
middlePara{1} = V' * V;
middlePara{2} = V' * (labelsPaired + C .* E);
middlePara{3} = G' * B;
middlePara{4} = ones(1, nPaired) * B;
middlePara{5} = B' * V;
middlePara{6} = G' * V;
middlePara{7} = ones(1, nPaired) * V;
middlePara{8} = X' * X;
middlePara{9} = Y' * Y;
middlePara{10} = X' * [B; BI];
middlePara{11} = Y' * [B; BT];
middlePara{12} = GI' * BI;
middlePara{13} = ones(1, nImageUnpaired) * BI;
middlePara{14} = GT' * BT;
middlePara{15} = ones(1, nTextUnpaired) * BT;
middlePara{16} = G' * V;
middlePara{17} = ones(1, nPaired) * V;
middlePara{18} = G' * V;
middlePara{19} = ones(1, nPaired) * V;
middlePara{20} = GT' * BT;
middlePara{21} = ones(1, nTextUnpaired) * BT;
middlePara{22} = GI' * BI;
middlePara{23} = ones(1, nImageUnpaired) * BI;
middlePara{24} = R;
middlePara{25} = P;
end

function DsPH_validate_batch(imagePaired, textPaired, imageUnpaired, textUnpaired, ...
    labelsPaired, labelsUnpairedImage, labelsUnpairedText, params)
if size(imagePaired, 1) ~= size(textPaired, 1) || ...
        size(imagePaired, 1) ~= size(labelsPaired, 1)
    error('DsPH:PairedSizeMismatch', 'Paired features and labels must align.');
end
if size(imageUnpaired, 1) ~= size(labelsUnpairedImage, 1) || ...
        size(textUnpaired, 1) ~= size(labelsUnpairedText, 1)
    error('DsPH:UnpairedSizeMismatch', 'Unpaired features and labels must align.');
end
if size(labelsPaired, 2) ~= size(labelsUnpairedImage, 2) || ...
        size(labelsPaired, 2) ~= size(labelsUnpairedText, 2)
    error('DsPH:LabelDimensionMismatch', 'All label matrices must have equal width.');
end
if size(labelsPaired, 1) < params.nbits
    error('DsPH:TooFewPairedSamples', 'The paired batch must contain at least nbits samples.');
end
end