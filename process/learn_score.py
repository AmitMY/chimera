from scorer.global_direction import GlobalDirectionExpert
from scorer.relation_direction import RelationDirectionExpert
from scorer.relation_transitions import RelationTransitionsExpert

from scorer.scorer import ProductOfExperts
from scorer.splitting_tendencies import SplittingTendenciesExpert
from utils.pipeline import Pipeline

LearnScorePipeline = Pipeline()
LearnScorePipeline.enqueue("plans", "Get Good Plans", lambda _, x: [p for g, p, s in x["pre-process"]["train"].data])
LearnScorePipeline.enqueue("relation-direction", "Learn Relation Direction",
                           lambda f, _: RelationDirectionExpert(f["plans"]))
LearnScorePipeline.enqueue("global-direction", "Learn Global Direction",
                           lambda f, _: GlobalDirectionExpert(f["plans"]))
LearnScorePipeline.enqueue("splitting-tendencies", "Learn Splitting Tendencies",
                           lambda f, _: SplittingTendenciesExpert(f["plans"]))
LearnScorePipeline.enqueue("relation-transitions", "Learn Relation Transitions",
                           lambda f, _: RelationTransitionsExpert(f["plans"]))
LearnScorePipeline.enqueue("scorer", "Create product of experts", lambda f, _: ProductOfExperts([
    f["relation-direction"],
    f["global-direction"],
    f["splitting-tendencies"],
    f["relation-transitions"],
]))
LearnScorePipeline.enqueue("out", "Expose the scorer", lambda f, _: f["scorer"])
