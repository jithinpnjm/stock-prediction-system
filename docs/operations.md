# Operations

## Local development

Install the environment with the dev extra and run:

    make lint
    make test

## Data

Fyers credentials must remain local:

    secrets/fyers_auth.json

or:

    FYERS_CLIENT_ID
    FYERS_ACCESS_TOKEN

Never commit real credentials.

## Expensive workloads

Full training, CPCV and GPU experiments should be manual CI/remote jobs. Cheap lint, unit tests and leakage tests run on every pull request.
