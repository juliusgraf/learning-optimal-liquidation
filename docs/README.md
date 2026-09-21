# Research documentation

## Model and operation

- [Model specification](model.md)
- [Revised auction clearing and separate rerun commands](revised_clearing_reruns.md)
- [Learning design](rl_design.md) and [continuous-control projection](continuous_action_extension.md)
- [Metrics schema](metrics_schema.md) and [research outputs](research_outputs.md)
- [Reproduction guide](../REPRODUCING.md) and [data setup](../data/README.md)

## Completed v20 study

- [Paper exhibits and saved numerical inputs](../README.md#paper-exhibits-and-their-sources)
- [Campaign configurations, seeds and selection records](../release/evidence/campaign.json)
- [Rough-Heston refinement](rough_heston_refinement/README.md)
- [Original v20 replacement recipe](rough_heston_refinement/rerun_v20.md)
- [Source and artifact provenance](../release/provenance.json)

The paper uses 320 refined synthetic runs and 200 retained historical runs.
Full run trees and checkpoints are not bundled. Manuscript sources, the top-level
development audit, legacy implementations and v19 documentation history were
removed from this branch; earlier Git commits are unchanged.
