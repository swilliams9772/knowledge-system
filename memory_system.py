class HierarchicalMemory:
    def __init__(self):
        self.short_term = VectorStore()  # Recent interactions
        self.episodic = ExperienceStore() # Past successful executions
        self.semantic = KnowledgeGraph()  # Domain knowledge
        
        # New memory types
        self.sensory = SensoryMemory(retention_seconds=30)
        self.working = WorkingMemory(max_items=7)
        self.episodic_memory = EpisodicMemory()
        
    def store(self, step, reasoning, result):
        # Store in short-term memory
        self.short_term.add(step, reasoning, result)
        
        # Store in sensory memory
        self.sensory.add({
            'step': step,
            'reasoning': reasoning,
            'result': result
        })
        
        # Store important items in working memory
        importance = self._calculate_importance(step, reasoning, result)
        if importance > 0.5:  # Threshold for importance
            self.working.add({
                'step': step,
                'reasoning': reasoning,
                'result': result
            }, importance)
        
        # If successful, store in episodic memories
        if result.success:
            self.episodic.add_experience(step, reasoning, result)
            self.episodic_memory.store_episode({
                'step': step,
                'reasoning': reasoning,
                'result': result
            })
            
        # Update knowledge graph
        self.semantic.update(step, reasoning, result)
        
    def _calculate_importance(self, step, reasoning, result) -> float:
        """Calculate importance score for working memory"""
        # Implementation depends on your specific needs
        # Example: higher importance for successful results
        base_score = 0.5
        if result.success:
            base_score += 0.3
        if hasattr(result, 'confidence'):
            base_score += result.confidence * 0.2
        return min(base_score, 1.0)