from typing import Dict, Any, List, Optional
from planning_strategies import (
    PlanningStrategy, 
    HierarchicalPlanner,
    IterativePlanner, 
    MonteCarloPlanner,
    ConstraintPlanner,
    CollaborativePlanner
)

class ExperienceAugmentedPlanner:
    def __init__(self):
        self.llm = OpenAI(temperature=0.2)
        
        # Initialize planning strategies
        self.strategies = {
            PlanningStrategy.HIERARCHICAL: HierarchicalPlanner(self.llm),
            PlanningStrategy.ITERATIVE: IterativePlanner(),
            PlanningStrategy.MONTE_CARLO: MonteCarloPlanner(),
            PlanningStrategy.CONSTRAINT: ConstraintPlanner(),
            PlanningStrategy.COLLABORATIVE: CollaborativePlanner()
        }
        
        self.strategy_selector = self._create_strategy_selector()
        
    def create_plan(self, query: str, context: Dict[str, Any], 
                   past_experience: Any) -> Dict[str, Any]:
        """Create plan using optimal strategy"""
        # Select best planning strategy
        strategy = self._select_strategy(query, context, past_experience)
        
        # Augment context with relevant past experiences
        augmented_context = self._augment_context(context, past_experience)
        
        # Generate initial plan using selected strategy
        initial_plan = self.strategies[strategy].plan(query, augmented_context)
        
        # Refine plan using iterative improvement
        if strategy != PlanningStrategy.ITERATIVE:
            refined_plan = self.strategies[PlanningStrategy.ITERATIVE].plan(
                initial_plan, 
                augmented_context
            )
        else:
            refined_plan = initial_plan
            
        return {
            'plan': refined_plan,
            'strategy': strategy.value,
            'context': augmented_context
        }
        
    def _select_strategy(self, query: str, context: Dict[str, Any], 
                        past_experience: Any) -> PlanningStrategy:
        """Select optimal planning strategy based on task characteristics"""
        features = self._extract_planning_features(query, context)
        
        # Use strategy selector to choose optimal strategy
        return self.strategy_selector.select(features)
        
    def _augment_context(self, context: Dict[str, Any], 
                        past_experience: Any) -> Dict[str, Any]:
        """Augment context with relevant past experiences"""
        augmented = context.copy()
        
        # Find relevant past experiences
        relevant_exp = past_experience.find_similar(context)
        
        # Extract useful patterns and insights
        patterns = self._extract_patterns(relevant_exp)
        
        # Add to context
        augmented['patterns'] = patterns
        augmented['relevant_experience'] = relevant_exp
        
        return augmented