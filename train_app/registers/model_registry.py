class ModelRegistry:
    def __init__(self):
        self.registered_models = {}

    def register(self, name):
        def decorator(cls):
            self.registered_models[name] = cls
            return cls
        return decorator
    
model_registry = ModelRegistry()
