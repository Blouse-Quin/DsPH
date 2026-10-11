function metrics = DsPH_evaluate(model, imageTest, textTest, labelsTest, topK)
% Evaluate image-to-text and text-to-image retrieval.

if nargin < 5 || isempty(topK)
    topK = min(100, size(model.codesImage, 1));
end
topK = min(topK, size(model.codesImage, 1));
imageTest = double(imageTest) - model.featureMeanImage;
textTest = double(textTest) - model.featureMeanText;

queryImage = DsPH_sign(imageTest * model.imageMap) > 0;
queryText = DsPH_sign(textTest * model.textMap) > 0;
trainImage = DsPH_compact_bits(model.codesImage > 0);
trainText = DsPH_compact_bits(model.codesText > 0);
testImage = DsPH_compact_bits(queryImage);
testText = DsPH_compact_bits(queryText);

distanceIT = DsPH_hamming_distance(testImage, trainText);
distanceTI = DsPH_hamming_distance(testText, trainImage);
truthIT = labelsTest * model.labelsText' > 0;
truthTI = labelsTest * model.labelsImage' > 0;

metrics.mapIT = DsPH_mean_average_precision(truthIT, distanceIT);
metrics.mapTI = DsPH_mean_average_precision(truthTI, distanceTI);
metrics.precisionAtK_IT = DsPH_precision_at_k(truthIT, distanceIT, topK);
metrics.precisionAtK_TI = DsPH_precision_at_k(truthTI, distanceTI, topK);
metrics.topK = topK;
end

function value = DsPH_mean_average_precision(truth, distances)
[~, order] = sort(distances, 2, 'ascend');
averagePrecision = zeros(size(truth, 1), 1);
validQueries = false(size(truth, 1), 1);
for queryIndex = 1:size(truth, 1)
    relevant = truth(queryIndex, order(queryIndex, :));
    relevantCount = sum(relevant);
    if relevantCount == 0
        continue;
    end
    validQueries(queryIndex) = true;
    cumulative = cumsum(relevant);
    precision = cumulative ./ (1:numel(relevant));
    averagePrecision(queryIndex) = sum(precision .* relevant) / relevantCount;
end
if any(validQueries)
    value = mean(averagePrecision(validQueries));
else
    value = 0;
end
end

function value = DsPH_precision_at_k(truth, distances, topK)
[~, order] = sort(distances, 2, 'ascend');
retrieved = truth(sub2ind(size(truth), ...
    repmat((1:size(truth, 1))', 1, topK), order(:, 1:topK)));
value = mean(sum(retrieved, 2) / topK);
end