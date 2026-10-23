from classes import ModelParams, PlaningParam, Model


COARSE_ALPHA = PlaningParam(0.20, 1.00, 0.20)
COARSE_GAMMA = PlaningParam(0.00, 1.00, 0.20)
COARSE_EPSILON = PlaningParam(0.00, 1.00, 0.20)

FINE_PARAM_MIN_OFFSET, FINE_PARAM_MAX_OFFSET, FINE_PARAM_INCREMENT = [0.10, 0.10, 0.02]


def plan_models_from_params(alpha: PlaningParam, gamma: PlaningParam, epsilon: PlaningParam) -> list[ModelParams]:
    planned_models = []

    while alpha.value <= alpha.max:
        while gamma.value <= gamma.max:
            while epsilon.value <= epsilon.max:
                planned_models.append(ModelParams(alpha.value, gamma.value, epsilon.value))
                epsilon.increment()
                
            epsilon.reset()
            gamma.increment()
            
        gamma.reset()
        alpha.increment()
        
    return planned_models


def calculate_coarse_models_plan() -> list[ModelParams]:
    return plan_models_from_params(COARSE_ALPHA, COARSE_GAMMA, COARSE_EPSILON)


def calculate_fine_models_plan(coarse_models: list[Model], models_to_refine: int) -> list[ModelParams]:
    coarse_models.sort(key=lambda model: model.results.winrate, reverse=True)
    best_course_models = coarse_models[:models_to_refine]
    
    planned_models = []
    
    def get_min_param(param: float) -> float:
        return max(param - FINE_PARAM_MIN_OFFSET, 0)
    
    def get_max_param(param: float) -> float:
        return min(param + FINE_PARAM_MAX_OFFSET, 1) 
    
    for coarse_model in best_course_models:
        alpha = PlaningParam(get_min_param(coarse_model.alpha), get_max_param(coarse_model.alpha), FINE_PARAM_INCREMENT)
        gamma = PlaningParam(get_min_param(coarse_model.gamma), get_max_param(coarse_model.gamma), FINE_PARAM_INCREMENT)
        epsilon = PlaningParam(get_min_param(coarse_model.epsilon), get_max_param(coarse_model.epsilon), FINE_PARAM_INCREMENT)
        
        planned_models = list(set(planned_models + plan_models_from_params(alpha, gamma, epsilon)))
        
    return planned_models