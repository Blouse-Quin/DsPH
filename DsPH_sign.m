function values = DsPH_sign(values)
% Use -1 rather than zero so every bit is binary.

values = sign(values);
values(values == 0) = -1;
end