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

1. New **model** logic: new `MODEL_REGISTRY` which stores the available model classes
    and new `build_model` function, which takes the model_name param directly from args
    and creates the model class with the args filtered by its signature.

    There is one single model (GNN4CD_model) independently on the loss functions. The only
    difference is in the `output_dims` param, which is now passed directly in the args.
    The model outputs a single tensor, with shape (n_nodes_batch, `output_dims`), which 
    contains the raw model output. This is then passes either to:
    - the loss function, which internally handles the transformation from raw model output
        to the desired quantities before applying the loss equation
    - an extractor function, which returns the raw model prediction (tranformed as the target is)
        In this case, the raw model prediction is then inverse transformed to get the actual
        precipitation or tasmax prediction

2. New **loss** logic: new LOSS_REGISTRY which stores the available loss classes
    and new build_loss function, which takes the loss_fn param directly from args
    and creates the loss class with the args filtered by its signature


