class TrainingParams:
    def __init__(self, training_steps: int, testing_steps: int, testing_rounds: int, output_dir: str):
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.testing_rounds = testing_rounds
        self.output_dir = output_dir
        
    def build_model_filepath(self, model_name: str) -> str:
        return self.output_dir + model_name + '.json'


class PlaningParam:
    def __init__(self, min: float, max: float, increment_amount: float) -> None:
        self.min = min
        self.max = max
        self.increment_amount = increment_amount
        self.value = min
        
    def increment(self):
        self.value += self.increment_amount
        
    def reset(self):
        self.value = self.min


class ModelParams:
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.formatted_name = f"a{f"{alpha:.2f}"}-g{f"{gamma:.2f}"}-e{f"{epsilon:.2f}"}".replace('.', '')


class TestResults:
    def __init__(self, won: int, lost: int, played: int, winrate: float, accuracy: float):
        self.won = won
        self.lost = lost
        self.played = played
        self.winrate = winrate
        self.accuracy = accuracy


class Model:
    def __init__(self, params: ModelParams, training_params: TrainingParams, test_results: TestResults):
        self.alpha = params.alpha
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.training = training_params
        self.results = test_results
        
        self.csv_dict = {
            'Alpha': f"{params.alpha:.3f}".replace('.', ','),
            'Gamma': f"{params.gamma:.3f}".replace('.', ','),
            'Epsilon': f"{params.epsilon:.3f}".replace('.', ','),
            'Points won': test_results.won,
            'Points lost': test_results.lost,
            'Points played': test_results.played,
            'Winrate %': f"{test_results.winrate:.3f}".replace('.', ','),
            'Accuracy %': f"{test_results.accuracy:.3f}".replace('.', ','),
            'Training steps': training_params.training_steps,
            'Testing steps': training_params.testing_steps,
            'Testing rounds': training_params.testing_rounds
        }