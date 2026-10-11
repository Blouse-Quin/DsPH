# DsPH

This folder contains a self-contained MATLAB implementation of the method
stored in the original project as `SPIH` and
`mymodel_2_maxmargin_B2B3`.

The implementation supports partially paired image-text data:

- paired samples have an image, a text feature, and a shared label matrix;
- unpaired image and text samples may have different label matrices;
- `DsPH_train` trains the first batch;
- `DsPH_online` updates the model with later batches.

## Quick start with MIRFlickr

From the project root:

```matlab
addpath('DsPH');
result = DsPH_demo;
```

The demo reads `data/MIRFlickr.mat`, which is already part of this project.
The expected variables are the complete matrices `Images`, `Texts`, and
`Labels`. The demo first reserves 5000 non-repeating test samples for ten
stream steps, then samples 2000 training samples with replacement per step
from the remaining data.

## Public API

```matlab
[model, batch] = DsPH_train(I_paired, T_paired, I_unpaired, T_unpaired, ...
    L_paired, L_unpaired_image, L_unpaired_text, params);

[model, batch] = DsPH_online(model, I_paired, T_paired, I_unpaired, ...
    T_unpaired, L_paired, L_unpaired_image, L_unpaired_text);

metrics = DsPH_evaluate(model, I_test, T_test, L_test);
```

For a later stream batch, call `DsPH_online` with the same seven data inputs
used by `DsPH_train`. The returned model keeps the accumulated codes, labels,
feature maps, and sufficient statistics for the next batch.

Each sample is one row. Image and text feature dimensions may differ. Labels
are rows aligned with their corresponding samples.

The default MIRFlickr parameters are returned by:

```matlab
params = DsPH_default_params('MIRFlickr', 32);
```

## Files

- `DsPH_train.m`: public first-batch training API.
- `DsPH_online.m`: public online update API.
- `DsPH_start.m`: first-batch optimization.
- `DsPH_online_update.m`: online sufficient-statistics update.
- `DsPH_evaluate.m`: image-to-text and text-to-image retrieval metrics.
- `DsPH_demo.m`: MIRFlickr example.

The optimization is extracted from the former
`UCMH/old_method/mymodel_2_maxmargin_B2B3` implementation. Historical result
files use either `mymodel_2_maxmargin_B2B3` or `SPIH` as the method name.