from model_training import train_models_from_plan
from model_planning import calculate_fine_models_plan
from classes import TrainingParams, ModelParams, TestResults, Model
import csv
from main import save_models_to_csv


def model_params_from_csv_dict(model_dict) -> ModelParams:
    return ModelParams(
        float(model_dict['Alpha'].replace(',', '.')),
        float(model_dict['Gamma'].replace(',', '.')),
        float(model_dict['Epsilon'].replace(',', '.'))
    )

def test_results_from_csv_dict(model_dict) -> TestResults:
    return TestResults(
        int(model_dict['Points won']),
        int(model_dict['Points lost']),
        int(model_dict['Points played']),
        float(model_dict['Winrate %'].replace(',', '.')),
        float(model_dict['Accuracy %'].replace(',', '.'))
    )


if __name__ == '__main__':
    with open('dokumenter/fine-resultater.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        all_fine_models = []
        
        for row in reader:
            model_params = model_params_from_csv_dict(row)
            test_results = test_results_from_csv_dict(row)
            training_params = TrainingParams(0, 0, 0, '')
            
            all_fine_models.append(Model(model_params, training_params, test_results))
    
    all_fine_models.sort(key=lambda model: model.results.winrate, reverse=True)
    best_fine_models = all_fine_models[:25]
    
    final_models_plan = [ModelParams(model.alpha, model.gamma, model.epsilon) for model in best_fine_models]
    final_models = train_models_from_plan(final_models_plan, TrainingParams(1000000, 100000, 25, 'bulk-trening/final-modeller/'))
    save_models_to_csv(final_models, 'dokumenter/final-resultater.csv')