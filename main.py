import csv
from classes import TrainingParams, Model, ModelParams
from model_planning import calculate_coarse_models_plan, calculate_fine_models_plan
from model_training import train_models_from_plan


COARSE_MODELS_OUTPUT_DIR = 'modeller/coarse/'
FINE_MODELS_OUTPUT_DIR =   'modeller/fine/'
FINAL_MODELS_OUTPUT_DIR =  'modeller/final/'

COARSE_RESULTS_OUTPUT_PATH = 'dokumenter/coarse-resultater.csv'
FINE_RESULTS_OUTPUT_PATH =   'dokumenter/fine-resultater.csv'
FINAL_RESULTS_OUTPUT_PATH =  'dokumenter/final-resultater.csv'


def save_models_to_csv(models: list[Model], output_filepath: str) -> None:
    with open(output_filepath, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = models[0].csv_dict.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(model.csv_dict for model in models)


if __name__ == '__main__':
    # Stage 1 
    coarse_models_plan = calculate_coarse_models_plan()
    coarse_models = train_models_from_plan(coarse_models_plan, TrainingParams(300000, 12500, 5, COARSE_MODELS_OUTPUT_DIR))
    save_models_to_csv(coarse_models, COARSE_RESULTS_OUTPUT_PATH)
    
    # Stage 2
    # Train all combinations (in increments of 0.02) around the best models from stage 1
    fine_models_plan = calculate_fine_models_plan(coarse_models, 8)
    fine_models = train_models_from_plan(fine_models_plan, TrainingParams(1000000, 25000, 10, FINE_MODELS_OUTPUT_DIR))
    save_models_to_csv(fine_models, FINE_RESULTS_OUTPUT_PATH)
    
    # Stage 3
    # Pick out and train the best from stage 2
    fine_models.sort(key=lambda model: model.results.winrate, reverse=True)
    best_fine_models = fine_models[:25]
    
    final_models_plan = [ModelParams(model.alpha, model.gamma, model.epsilon) for model in best_fine_models]
    final_models = train_models_from_plan(final_models_plan, TrainingParams(1000000, 1000000, 25, FINAL_MODELS_OUTPUT_DIR))
    save_models_to_csv(final_models, FINAL_RESULTS_OUTPUT_PATH)