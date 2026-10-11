function [model, batch] = DsPH_online(model, imagePaired, textPaired, ...
    imageUnpaired, textUnpaired, labelsPaired, labelsUnpairedImage, labelsUnpairedText)
% Update a trained DsPH model with one later batch.

imagePaired = double(imagePaired) - model.featureMeanImage;
imageUnpaired = double(imageUnpaired) - model.featureMeanImage;
textPaired = double(textPaired) - model.featureMeanText;
textUnpaired = double(textUnpaired) - model.featureMeanText;
[codeImage, codeText, imageMap, textMap, middlePara, info] = DsPH_online_update(...
    imagePaired, textPaired, imageUnpaired, textUnpaired, double(labelsPaired), ...
    double(labelsUnpairedImage), double(labelsUnpairedText), model.params, model.middlePara);

model.imageMap = imageMap;
model.textMap = textMap;
model.codesImage = [model.codesImage; codeImage];
model.codesText = [model.codesText; codeText];
model.labelsImage = [model.labelsImage; labelsPaired; labelsUnpairedImage];
model.labelsText = [model.labelsText; labelsPaired; labelsUnpairedText];
model.middlePara = middlePara;
model.info = info;
batch.codeImage = codeImage;
batch.codeText = codeText;
batch.info = info;
end