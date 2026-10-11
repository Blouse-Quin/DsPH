function [trainStream, testStream] = DsPH_split_stream(imageData, textData, labels, ...
    trainN, testN, batchN)
% Reserve a non-repeating test stream, then sample training batches with replacement.

if size(imageData, 1) ~= size(textData, 1) || ...
        size(imageData, 1) ~= size(labels, 1)
    error('DsPH:DataSizeMismatch', 'Image, text, and label rows must align.');
end
testTotal = batchN * testN;
if size(imageData, 1) < testTotal
    error('DsPH:NotEnoughData', ...
        'Need %d samples for the test stream, but only %d are available.', ...
        testTotal, size(imageData, 1));
end

allIndex = randperm(size(imageData, 1));
testIndex = allIndex(1:testTotal);
trainPool = allIndex(testTotal + 1:end);
if isempty(trainPool) && trainN > 0
    error('DsPH:NoTrainingData', 'No samples remain after reserving the test stream.');
end
trainStream = cell(1, batchN);
testStream = cell(1, batchN);
for batchIndex = 1:batchN
    trainIndex = trainPool(randi(numel(trainPool), 1, trainN));
    batchTestStart = (batchIndex - 1) * testN + 1;
    batchTestIndex = testIndex(batchTestStart:batchTestStart + testN - 1);
    trainStream{batchIndex}.image = imageData(trainIndex, :);
    trainStream{batchIndex}.text = textData(trainIndex, :);
    trainStream{batchIndex}.labels = labels(trainIndex, :);
    testStream{batchIndex}.image = imageData(batchTestIndex, :);
    testStream{batchIndex}.text = textData(batchTestIndex, :);
    testStream{batchIndex}.labels = labels(batchTestIndex, :);
end
end