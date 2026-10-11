function normalized = DsPH_normalize_rows(values)
% Normalize each row without changing the matrix shape.

rowNorm = sqrt(sum(values.^2, 2));
rowNorm(rowNorm < 1e-14) = 1;
normalized = values ./ rowNorm;
end