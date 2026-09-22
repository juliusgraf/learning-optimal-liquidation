# Contributing

Follow the [README installation steps](README.md#installation), including the
CPU-only Torch preinstallation on Linux, to install the development dependencies.
Run the offline suite before submitting a change:

```bash
pytest -q -m 'not network and not slow and not market_data'
```

Historical integration tests require separately obtained market data and skip
when those inputs are absent. With the research data installed, use
`pytest --require-market-data -q -m 'not network and not slow'` to require the
complete offline suite. Synthetic and generated-fixture tests need no credentials.

Explain the problem, the change and relevant validation. Changes to rewards,
state information, action projection, timing, clearing or data selection change
the experimental contract and must be documented. Update the artifact contract
when necessary; do not silently reuse incompatible checkpoints.

Preserve independent train/validation/test streams, economic checkpoint selection,
and complete paired seed reporting. Keep raw data, credentials, personal tool
settings, manuscript drafts and generated run outputs out of contributions.
Never overwrite completed campaigns. Reporting changes should consume saved
records and should not launch training or step the environment.

The repository distributes code and compact research evidence. The MIT license
does not grant rights to third-party market data.
