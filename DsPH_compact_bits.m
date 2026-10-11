function compact = DsPH_compact_bits(bits)
% Pack binary columns into uint8 words for fast Hamming distance.

[nSamples, nBits] = size(bits);
nWords = ceil(nBits / 8);
compact = zeros(nSamples, nWords, 'uint8');
for bitIndex = 1:nBits
    wordIndex = ceil(bitIndex / 8);
    compact(:, wordIndex) = bitset(compact(:, wordIndex), ...
        mod(bitIndex - 1, 8) + 1, bits(:, bitIndex));
end
end