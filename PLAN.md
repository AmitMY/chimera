# Sentence Planning from Graphs - Methods and Evaluation
In previous work we divided graph2text into 2 tasks:
- Text Planning - `p(g)`
- Surface realization - `r(p(g))`

It is simple to think on an evaluation methodology for the entire system, for example by comparing it fully to any other system, but it is tricky to evaluate each component.
To evaluating the surface realization component, instead of running `r(p(g))` we can take the correct text plan and realize it - `r(ref-p)`, evaluating with BLEU or something else. 
However, to evaluate text planning we shouldn't compare `r(p1(g))` to `r(p2(g))` as we can't rely on the quality of the transformation of `r` to be correlative to the quality of the plans.

## Planning Methods
### Product-of-Experts Naive Planner
This planner first generates all possible plans, then ranks them using a uniform product of experts as described in the previous paper.
It is naive because it uses a uniform product.

### Neural Planner
The neural planner is mimicking the user process for graph traversal, in the following steps: (initializing `nodes_stack=[]`)

1. If `nodes_stack` is empty, select a node to start with (`nodes_stack.push(node)`). Imaging a person putting their finger on the node.
2. If `nodes_stack` is not empty, choose from the following:
    1. All undirected edges from that node - imagine their finger going to the next node  (`nodes_stack.push(next_node)`)
    2. If already expressed at least 1 edge, `pop` (`nodes_stack.pop()`) - imagine their finger going back on the last edge they traversed. If they pop and `len(nodes_stack) == 0`, lift the hand
    
At any point where the model has multiple choices we put a loss on the reference choice.

Every node and edge are represented by their environment, but I won't go into too many details here.

This is stupid fast! The performance with small vectors (size 10) is similar to large ones (size 300).

This also tackles the problem of seq2seq, as the network has a really hard time decoding structure, unless we only give it these options.

## Evaluating Text Planning
Ideally, we want a planner that produces plans exactly like a human does. However, humans are complex, and the reasons for choosing one sentence plan over another are often arbitrary.
We do note that some sets of plans are more plausible than others, and this is what we will try to evaluate.

If we generate every possible plan, and give it a likelihood score we can rank all plans from most plausible to least.
A good planning algorithm would rank the reference plan as close to #1 as possible. We can show that by translating "good" plans compared to not-so-good, and see a drop in evaluation performance.

#### Methodically:
1. Generate all plans
2. Rank them by scores
    1. NaivePlanner - by product of experts
    2. NeuralPlanner - log-likelihood of that plan
3. Find the rank of the reference plan normalized by the amount of possible plans.
4. The total score of the planner is the average of all normalized ranks.

We can use this methodology to improve our planners as well. The neural planner can instead of taking the last model weights, take the best dev weights on this ranking score,
while the naive planner can learn power weights for each expert, as currently they are uniform, which can't be the best configuration. This will also allow adding many more experts, and figuring out if they are good or not.

### Technical Difficulties
It can take many hours to create all plans for the training data, and TBs of memory and storage. It will then take days or weeks to evaluate all plans for their likelihood.

#### Mitigation Heuristics
- We can limit our graph sizes to 4 or 5 edges instead of 7, which will decrease the amount of time and memory by a few orders of magnitude.
- We can choose a sufficely large constant number of random plans (e.g. 1000) and add the reference plan. This will give an approximation to the performance on the entire set.
