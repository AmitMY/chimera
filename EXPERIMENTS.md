# Experiments
This file contains experiments the authors are currently making.

## Speed Compared to Score
If we try our neural planner, we get:
- Training time: 60 seconds (can easily be 30 seconds or less)
- Test time (1 cpu): 3 seconds
- WebNLG - BLEU [45.79, 79.0, 54.4, 38.5, 28.0] (20,000 steps m1)
- WebNLG - BLEU [46.35, 78.1, 54.5, 38.7, 28.1] (40,000 steps m2)
- Delexicalized WebNLG - BLEU [44.21, 81.7, 56.4, 39.5, 28.2] (40,000 steps m3)
- Delexicalized WebNLG - BLEU [44.46, 80.1, 55.1, 38.3, 27.1] (40,000 steps m4)

Compared to the following using our naive product of experts planner, with the same realizer snapshots:
- Training time: 0 seconds
- Test time (1 cpu): 5300 seconds
- Test time (40 cpus): 450 seconds
- WebNLG - BLEU [45.19, 77.8, 53.2, 37.4, 27.0] (20,000 steps m1)
- WebNLG - BLEU [45.49, 77.9, 53.7, 37.8, 27.1] (40,000 steps m2)
- Delexicalized WebNLG - BLEU [44.77, 79.3, 53.7, 37.4, 26.4] (40,000 steps m3)
- Delexicalized WebNLG - BLEU [46.01, 79.8, 55.0, 38.3, 27.2] (40,000 steps m4)


## Neural Planner Comparison

Greedy best neural plan: 
- BLEU [45.97, 77.8, 54.1, 38.4, 28.0]
Sample neural:
- BLEU [43.76, 77.1, 51.9, 35.9, 25.5]
- BLEU [43.74, 76.9, 51.9, 35.9, 25.5]
- BLEU [44.7, 77.9, 52.7, 36.8, 26.4]
- BLEU [44.64, 77.9, 52.7, 36.8, 26.3]
- BLEU [44.12, 77.5, 52.2, 36.2, 25.9]


Combined model:
Sample 0 + Greedy = BLEU [45.88, 77.7, 54.0, 38.2, 27.7]
Sample 1 + Greedy = BLEU [45.35, 78.7, 54.0, 38.1, 27.5]
Sample 5 + Greedy = BLEU [45.87, 78.6, 54.1, 38.1, 27.6]
Sample 50 + Greedy = BLEU [45.58, 78.2, 53.7, 37.7, 27.2]



beam=5
BLEU 45.46
+ check all entities are covered
BLEU 45.62

beam=10
BLEU 45.81
+check
BLEU 46.06

beam=50
BLEU 45.97
+check
BLEU 46.41

