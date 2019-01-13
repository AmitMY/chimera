# Chimera

## Environment
We recommend installing all dependencies in a separate Conda environment.

### GPU-support
This code will run with or without a CUDA, but we recommend using a machine with CUDA.

### Dependencies
Execute `setup.sh`. This will install pip dependencies, as well as OpenNMT.

## TODO
- Pipeline class should load cache on the fly instead of preemptive.
- Pipeline executer should have progressbar support.
- Planning should be a separate interchangable class - Planner, which has `plan_best` and `plan_all` methods.

## Process
For training, the main pipeline consists of these sub-pipelines:
1. Preprocess Training (both train and dev sets)
    1. Load the data-set
    1. Convert RDFs to graphs
    1. Fix misspellings
    1. Locate entities in the text
    1. Match plans for each graph and reference
    1. Tokenize the plans and reference sentences   
1. Train Model
    1. Initialize model
    1. Pre-process training data
    1. Train Model
    1. Find best checkpoint, chart all checkpoints
1. Learn Score
    1. Get good plans from training set
    1. Learn Relation-Direction Expert
    1. Learn Global-Direction Expert
    1. Learn Splitting-Tendencies Expert
    1. Learn Relation-Transitions Expert
    1. Create Product of Experts
1. Preprocess Test Set
    1. Load the data-set
    1. Convert RDFs to graphs
    1. Fix misspellings
    1. Generate best plan
    1. Tokenize plans & sentences
1. Translate
    1. Translate test plans into text
    1. Post-process translated texts
    1. Save Translations to file (for human reference)
1. Evaluate model performance
    1. Evaluate test reader

Once running the main pipeline, every pipeline result is cached. 
If the cache is removed, the pipeline will continue from its last un-cached process.

**Note:** by default, all pipelines are muted, meaning any screen output will not present on screen.


## Example
Output running for the second time: (runs for just a few seconds to load the caches)

![Second Run Pipeline](git-assets/second-run.png)

The expected result (will show on screen) reported by `multi-bleu.perl` is around:

**BLEU [47.27, 79.6, 55.3, 39.4, 28.7]**

