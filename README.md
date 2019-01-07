# Chimera
## Dependencies
**We recommend installing all dependencies in a separate Conda environment.**

Execute `setup.sh`. This will install pip dependencies, as well as OpenNMT.

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
1. Learn Scores
    1. Learn Experts
        1. Learn 1st expert
        1. Learn 2nd expert
        1. Learn nth expert
1. Preprocess Test Set
    1. Load the data-set
    1. Convert RDFs to graphs
    1. Fix misspellings
1. Plan for Test Set
    1. Generate all plans for each graph
    1. Score all plans for each graph, return best scored
1. Translate
    1. Translate test plans into text
1. Post-process
    1. Referring expressions generation
1. Evaluate model performance

Once running the main pipeline, every pipeline result is cached. 
If the cache is removed, the pipeline will continue from its last un-cached process.

Note, by default, all pipelines are muted, meaning any screen output will not present on screen.

## Data
