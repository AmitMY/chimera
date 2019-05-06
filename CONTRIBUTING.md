# Contributing
In this repository we welcome contributions. 
If you found a problem, or want to add a feature, please create a Pull-Request or open an Issue.

## We need your help!
There are a few problems we haven't managed to solve on ourselves. You might be the one to solve them!

- In `NaivePlanner`, calling `plan_best` does not clear memory. Instead, if called in a loop will kill the process with an out-of-memory error.
