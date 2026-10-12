function params = DsPH_default_params(dataName, nBits)
% Return the parameter defaults used by the original DsPH experiments.

if nargin < 1 || isempty(dataName)
    dataName = 'MIRFlickr';
end
if nargin < 2 || isempty(nBits)
    nBits = 32;
end

params.alpha1 = 1;
params.alpha2 = 1;
params.beta1 = 10;
params.beta2 = 10;
params.beta3 = 0.1;
params.xi = 1;
params.nbits = nBits;
params.maxIter = 10;
params.epsilon = 1e-5;
params.verbose = true;
params.trainN = 2000;
params.testN = 500;
params.batchN = 10;
params.partialRate = 0.5;

switch lower(char(dataName))
    case 'mirflickr'
        % These values match the original MIRFlickr DsPH implementation.
    case 'nuswide'
        params.alpha1 = 10;
        params.alpha2 = 100000;
        params.beta1 = 100000;
        params.beta2 = 1000;
        params.beta3 = 0.001;
    case 'iaprtc-12'
        params.alpha1 = 0.0001;
        params.alpha2 = 0.0001;
        params.beta1 = 10000;
        params.beta2 = 10000;
        params.beta3 = 1;
    otherwise
        error('DsPH:UnknownDataset', 'Unknown dataset: %s', char(dataName));
end
end