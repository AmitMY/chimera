# Contributing
In this repository we welcome contributions. 
If you found a problem, or want to add a feature, please create a Pull-Request or open an Issue.

## We need your help!
There are a few problems we haven't managed to solve on ourselves. You might be the one to solve them!

- In `NaivePlanner`, calling `plan_best` does not clear memory. Instead, if called in a loop will kill the process with an out-of-memory error.
- In `OpenNMTModelRunner`, calling `train` returns all checkpoints, which then need to be evaluated. While this gives a lot of flexibility, evaluating every checkpoint is very slow, and we're looking for a trick to evaluate BLEU in training time.
- In `OpenNMTModelRunner`, calling train hides the stdout as intended, but it is hard to gauge the time it takes to train. It would be nice to parse the stdout and run a tqdm progress bar. 