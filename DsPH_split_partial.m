function batch = DsPH_split_partial(streamBatch, partialRate)
% Convert one fully paired batch into the old DsPH semi-paired layout.

sampleCount = size(streamBatch.image, 1);
unpairedCount = round(partialRate * sampleCount);
pairedCount = sampleCount - unpairedCount;
imageUnpairedCount = floor(unpairedCount / 2);
textUnpairedCount = unpairedCount - imageUnpairedCount;

pairedIndex = 1:pairedCount;
imageUnpairedIndex = sampleCount:-1:(sampleCount - imageUnpairedCount + 1);
textUnpairedIndex = pairedCount + 1:pairedCount + textUnpairedCount;

batch.imagePaired = streamBatch.image(pairedIndex, :);
batch.textPaired = streamBatch.text(pairedIndex, :);
batch.labelsPaired = streamBatch.labels(pairedIndex, :);
batch.imageUnpaired = streamBatch.image(imageUnpairedIndex, :);
batch.labelsUnpairedImage = streamBatch.labels(imageUnpairedIndex, :);
batch.textUnpaired = streamBatch.text(textUnpairedIndex, :);
batch.labelsUnpairedText = streamBatch.labels(textUnpairedIndex, :);
end