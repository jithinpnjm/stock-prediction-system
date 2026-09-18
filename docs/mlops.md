# MLOps

Every model run should be reproducible from:

- Git commit
- DVC dataset/version identity
- feature version
- label version
- split configuration
- random seed
- Python/package/runtime information
- hardware metadata
- model parameters
- OOF predictions

MLflow is the system of record for experiments and model artifacts.

Model registration is intentionally blocked unless the frozen holdout has been
evaluated and the caller explicitly enables registration.
