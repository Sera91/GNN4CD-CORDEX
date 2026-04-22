## GNN4CD-CORDEXML - branch Refactoring_Valentina

This folder contains the refactored code. Main modifications:

1. New **REGISTRY** logic, to improve the code modularity and facilitate the addition
    of new features without modifying the core code. The registries are listed below:
    - `MODEL_REGISTRY` in `models/registry.py`
    - `LOSS_REGISTRY` in `utils/losses/registry.py`
    - `EXTRACTOR_REGISTRY` in `utils/extractors.registry.py`
    - `TRANSFORM_PREDICTAND_REGISTRY` in `utils/predictand_transforms/registry.py`
    - `INVERSE_TRANSFORM_PREDICTAND_REGISTRY` in `utils/predictand_transforms/registry.py`
    - `TRANSFORM_PREDICTOR_REGISTRY` in `utils/predictor_transforms/registry.py`

2. New `models/build_model.py` and `utils/losses/build_loss.py` functions, which take the
    model_name and loss_fn params, respectively, directly from args and create the model
    and loss classes with the args automatically filtered by their signature.

3. Unified `GNN4CD_model` which works for all the available losses. The different output dimension
    (the only difference between the losses) is given in the new `args.output_dims` parameter.
    
    Specifically, we need:
    - `OUTPUT_DIMS = 1` for `MSE_QMSE_PSD_Loss`
    - `OUTPUT_DIMS = 3` for `BernoulliGammaNLLLoss`
    - `OUTPUT_DIMS = 2` for `GaussianNLLLoss`

    Now the model returns a single tensor of shape: (num_nodes, `output_dims`) and the loss functions
    have been modified to internally handle the transformation from raw model output to the desired
    quantities before applying the loss equation.

    For inference (validation, predictions) the extractor function is used. This function is specific to
    the loss function used to train the model and handles the transformation from raw model output to the desired
    precipitation or tasmax predictions.

4. Unified `utils/trainining/train.py` which handles the training loop, independently on the chosen model
    and loss function. This is possible thanks to the updated model and loss logics, as well as the new
    way to inverse transform the predictand and get the precipitation or tasmax predictions.


