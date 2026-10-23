import time
from concurrent.futures import ProcessPoolExecutor
from pong_ai_modified import train as train_model, test as evaluate_model
from classes import TrainingParams, ModelParams, Model


def train_and_evaluate_model(params: ModelParams, training_params: TrainingParams) -> Model:
    start_time = time.perf_counter()
    
    model_filepath = training_params.build_model_filepath(params.formatted_name)
    
    train_model(model_filepath, params.alpha, params.gamma, params.epsilon, training_params.training_steps)
    test_results = evaluate_model(model_filepath, training_params.testing_rounds, training_params.testing_steps)
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Trent og testet model: {params.formatted_name}. Tok {elapsed_time:.2f} sekunder.")
    
    return Model(params, training_params, test_results)


def train_models_from_plan(models_plan: list[ModelParams], training_params: TrainingParams) -> list[Model]:
    start_time = time.perf_counter()
    print(f"Trener {len(models_plan)} modeller.")    

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(train_and_evaluate_model, model_params, training_params)
            for model_params in models_plan
        ]
        results = [future.result() for future in futures]
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Trent {len(results)} modeller. Tok {elapsed_time:.2f} sekunder.")
    
    return results