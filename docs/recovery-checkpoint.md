# Recovery checkpoint

The research-grade rebuild is developed in small, durable Git commits.

The branch research-grade-rebuild is the recovery branch. Every material code or
configuration change should be committed before beginning a new logical workstream.

The previous timeout risk came from keeping a large final patch in memory before
committing it. The project now follows this checkpoint rule:

1. implement one logical layer;
2. inspect the resulting Git tree;
3. commit it;
4. only then continue.

Raw market data and credentials are never used as commit checkpoints: they remain
outside Git or under DVC/ignored paths as appropriate.
