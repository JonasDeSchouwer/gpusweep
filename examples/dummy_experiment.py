from configs import pydraclass, main
import numpy as np
import time

@pydraclass
class DummyExperimentConfig:
    name: str = "dummy_experiment"
    num_parameters: int = 10
    seed: int = 42

def run_experiment(config: DummyExperimentConfig):
    # Dummy experiment - returns a random number
    time.sleep(5)
    result = np.random.normal(0,1)
    return result

@main(DummyExperimentConfig)
def main(config: DummyExperimentConfig):
    run_experiment(config)
