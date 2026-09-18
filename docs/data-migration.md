# Safe DVC migration

The repository currently retains legacy market-data blobs so no historical data is lost.

Before removing those blobs from Git:

1. Configure a DVC remote outside committed configuration.
2. Add the canonical raw source and durable derived datasets with DVC.
3. Push the objects to the remote.
4. Clone into a clean directory and run DVC pull.
5. Verify row counts and SHA-256 hashes against the source.
6. Remove the large Git blobs only in a separate migration commit.

Never commit private SSH key paths, credentials, tokens, or passwords in DVC configuration.
