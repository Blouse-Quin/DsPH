function result = DsPH_demo()
% Run the original ten-time-step DsPH experiment on MIRFlickr.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataFile = fullfile(projectRoot, 'data', 'MIRFlickr.mat');
if ~isfile(dataFile)
    error('DsPH:MissingMIRFlickr', 'Expected data file: %s', dataFile);
end

data = load(dataFile);
rng(0, 'twister');
params = DsPH_default_params('MIRFlickr', 32);
if isfield(data, 'Images') && isfield(data, 'Texts') && isfield(data, 'Labels')
    imageData = data.Images;
    textData = data.Texts;
    labelData = data.Labels;
elseif isfield(data, 'I_tr') && isfield(data, 'T_tr') && isfield(data, 'L_tr')
    imageData = data.I_tr;
    textData = data.T_tr;
    labelData = data.L_tr;
    if isfield(data, 'I_te') && isfield(data, 'T_te') && isfield(data, 'L_te')
        imageData = [imageData; data.I_te];
        textData = [textData; data.T_te];
        labelData = [labelData; data.L_te];
    end
else
    error('DsPH:UnsupportedMIRFlickrFormat', ...
        'Expected Images/Texts/Labels or I_tr/T_tr/L_tr variables.');
end

testSampleCount = params.batchN * params.testN;
if size(imageData, 1) < testSampleCount + 1
    error('DsPH:NotEnoughMIRFlickrData', ...
    'Need at least %d samples to reserve the test stream.', testSampleCount);
end
[trainStream, testStream] = DsPH_split_stream(imageData, textData, labelData, ...
    params.trainN, params.testN, params.batchN);
result.mapIT = zeros(1, params.batchN);
result.mapTI = zeros(1, params.batchN);
result.precisionAtK_IT = zeros(1, params.batchN);
result.precisionAtK_TI = zeros(1, params.batchN);
result.trainInfo = cell(1, params.batchN);

for timeIndex = 1:params.batchN
    batch = DsPH_split_partial(trainStream{timeIndex}, params.partialRate);
    if timeIndex == 1
        [model, batchInfo] = DsPH_train(batch.imagePaired, batch.textPaired, ...
            batch.imageUnpaired, batch.textUnpaired, batch.labelsPaired, ...
            batch.labelsUnpairedImage, batch.labelsUnpairedText, params);
    else
        [model, batchInfo] = DsPH_online(model, batch.imagePaired, batch.textPaired, ...
            batch.imageUnpaired, batch.textUnpaired, batch.labelsPaired, ...
            batch.labelsUnpairedImage, batch.labelsUnpairedText);
    end

    metrics = DsPH_evaluate(model, testStream{timeIndex}.image, ...
        testStream{timeIndex}.text, testStream{timeIndex}.labels, 100);
    result.mapIT(timeIndex) = metrics.mapIT;
    result.mapTI(timeIndex) = metrics.mapTI;
    result.precisionAtK_IT(timeIndex) = metrics.precisionAtK_IT;
    result.precisionAtK_TI(timeIndex) = metrics.precisionAtK_TI;
    result.trainInfo{timeIndex} = batchInfo.info;
    fprintf('DsPH t=%d/%d: mAP I->T %.4f, T->I %.4f\n', ...
        timeIndex, params.batchN, metrics.mapIT, metrics.mapTI);
end

result.params = params;
result.model = model;
end