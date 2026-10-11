function [model, batch] = DsPH_train(imagePaired, textPaired, imageUnpaired, ...
    textUnpaired, labelsPaired, labelsUnpairedImage, labelsUnpairedText, params)
% Public first-batch training API for DsPH.

if nargin < 8 || isempty(params)
    params = DsPH_default_params('MIRFlickr', 32);
end
params = DsPH_complete_params(params, size(labelsPaired, 1));

imagePaired = double(imagePaired);
textPaired = double(textPaired);
imageUnpaired = double(imageUnpaired);
textUnpaired = double(textUnpaired);
labelsPaired = double(labelsPaired);
labelsUnpairedImage = double(labelsUnpairedImage);
labelsUnpairedText = double(labelsUnpairedText);

model.featureMeanImage = mean([imagePaired; imageUnpaired], 1);
model.featureMeanText = mean([textPaired; textUnpaired], 1);
imagePaired = imagePaired - model.featureMeanImage;
imageUnpaired = imageUnpaired - model.featureMeanImage;
textPaired = textPaired - model.featureMeanText;
textUnpaired = textUnpaired - model.featureMeanText;

[codeImage, codeText, imageMap, textMap, middlePara, info] = DsPH_start(...
    imagePaired, textPaired, imageUnpaired, textUnpaired, labelsPaired, ...
    labelsUnpairedImage, labelsUnpairedText, params);

model.params = params;
model.imageMap = imageMap;
model.textMap = textMap;
model.codesImage = codeImage;
model.codesText = codeText;
model.labelsImage = [labelsPaired; labelsUnpairedImage];
model.labelsText = [labelsPaired; labelsUnpairedText];
model.middlePara = middlePara;
model.info = info;
batch.codeImage = codeImage;
batch.codeText = codeText;
batch.info = info;
end

function params = DsPH_complete_params(params, pairedCount)
defaults = DsPH_default_params('MIRFlickr', 32);
names = fieldnames(defaults);
for nameIndex = 1:numel(names)
    name = names{nameIndex};
    if ~isfield(params, name) || isempty(params.(name))
        params.(name) = defaults.(name);
    end
end
if ~isfield(params, 'nbits') || isempty(params.nbits)
    params.nbits = min(32, pairedCount);
end
end