function codes = DsPH_dcc_update(codes, Q, H)
% Perform one sequential DCC sweep over code bits.

for bit = 1:size(codes, 2)
    otherBits = [1:bit - 1, bit + 1:size(codes, 2)];
    scores = Q(:, bit) - codes(:, otherBits) * H(otherBits, bit);
    updated = sign(scores);
    zeroScores = scores == 0;
    updated(zeroScores) = codes(zeroScores, bit);
    codes(:, bit) = updated;
end
end
