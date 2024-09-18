# EgoTracks

The following code was adapted from the [EgoTracks](https://github.com/EGO4D/episodic-memory/tree/main/EgoTracks) repository.

## Original Code
- **Repository:** [EgoTracks](https://github.com/EGO4D/episodic-memory/tree/main/EgoTracks)
- **Author:** [tanghaotommy](https://github.com/tanghaotommy)
- **Commit:** `497ba92` (Date: 01-03-2023)

## Modifications
- **Purpose of Modifications:** Removed references to `tracking.dataset.ego4d_tracking`.
- **Changes Made:**
  - Removed the import statement causing no module `No module named 'tracking.dataset.ego4d_tracking'` issue as described [here](https://github.com/EGO4D/episodic-memory/issues/51), i.e. `from tracking.dataset.train_datasets.ego4d_vq import Ego4DVQ` in [this file](./tracking/dataset/build.py).
  - Added a symbolic link to the tools directory in the tracking folder to avoid the build.
  - Updated code to print internal warnings only if the verbose parameter is enabled [here](./tracking/models/stark_tracker/resnet.py).