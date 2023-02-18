A pure Julia transcription of the hidden (state) conditional random field (__HCRF__) at https://github.com/althonos/pyhcrf.

## Upgrades

- multi-threaded learning phase
- option to calculate the most likely hidden state sequence
- supports input and fast learning on overlapping sequences
- added unconstrained transitions (now default)
- optimised internal data representation

## Fixes

- return probabilities when overflow are now correct
- removed repeated entries in step transitions
