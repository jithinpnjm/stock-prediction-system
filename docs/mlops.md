# MLOps

Git versions code and configuration. DVC versions datasets. MLflow versions experiments and model artifacts.

Every training run records:

- Git commit;
- dataset/feature/label/validation versions;
- resolved configuration;
- Python/runtime metadata;
- fold metrics;
- OOF predictions;
- model artifacts.

Model registration is non-destructive. No script changes a champion alias unless explicitly requested by the operator.
