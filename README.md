A pure Julia transcription of the hidden (state) conditional random field (__HCRF__) at https://github.com/althonos/pyhcrf.

## Upgrades

- multi-threaded learning phase
- option to calculate the most likely hidden state sequence
- supports input of and fast learning on overlapping sequences
- specialised, faster code path for sparse features
- added unconstrained transitions (now default)
- ability to start from existing model
- extended callback function to enable examining model state and performance during learning
- optimised internal data representation for these changes

## Fixes

- return probabilities when overflow are now correct
- removed repeated entries in step transitions
