# Local Search Algorithms for Neural Architecture Search in Super Resolution

## Introduction

This repository provides a suite of evaluation functions and local search algorithms designed for neural architecture search (NAS). This repository is flexible and can be integrated with any search space as long as it uses a binary or Gray code encoding.

The primary goal is to refine and optimize neural network architectures by exploring the search space locally, balancing multiple objectives:

- **Performance**: Measured by PSNR or SynFlow.
- **Model Complexity**: Number of parameters.
- **Computational Cost**: Measured in FLOPs.

---

## Features

- **Flexible Integration**: Can be used with any search space that has a binary or Gray code encoding.
- **Local Search Algorithms**: Includes implementations of Hill Climbing, Tabu Search, and Simulated Annealing.
- **Multi-Objective Evaluation**: Supports evaluation based on PSNR, SynFlow, parameter count, and FLOPs.
- **Customizable Metrics**: Easily integrate your own evaluation metrics.
- **Modular Design**: Organized code structure for easy understanding and modification.

---

## Performance Comparison

As part of the optimization process, the local search algorithms aim to find architectures that are locally optimal based on the defined objectives.

**Note**: Specific performance metrics will depend on your search space, datasets, and configurations. It's recommended to log and analyze results after running the algorithms.

---

## How It Works

### Integration with Search Spaces

- **Search Space Agnostic**: The repository is designed to be independent of any specific search space.
- **Encoding**: Requires architectures to be represented using binary or Gray code encoding.
- **Decoding Function**: Users need to provide their own decoding function to convert the binary/Gray code into a model architecture.

### Evaluation Metrics

- **PSNR**: Measures image reconstruction quality.
- **SynFlow**: Proxy metric for evaluating architectures without full training.
- **FLOPs and Parameters**: Assess computational cost and model complexity.
- **Custom Metrics**: You can integrate additional metrics as needed.

### Optimization Process

- **Local Search Algorithms**: Start with random architectures and iteratively refine them through local modifications.
  - **Hill Climbing**: Makes incremental changes, accepting improvements.
  - **Tabu Search**: Uses memory structures to avoid revisiting recently explored solutions.
  - **Simulated Annealing**: Explores the search space probabilistically, allowing occasional acceptance of worse solutions to escape local minima.

---

## Project Structure

```plaintext
Eval-LocalSearch/
├── config.py
├── evaluation.py
├── local_search.py
├── utils.py
├── main.py
├── README.md
├── LICENSE
└── requirements.txt
```

- **config.py**: Configuration settings, including evaluation metrics and local search parameters.
- **evaluation.py**: Evaluation functions for PSNR, SynFlow, FLOPs, and parameter counting.
- **local_search.py**: Implementation of local search algorithms.
- **utils.py**: Utility functions, such as dominance checks.
- **main.py**: Main script to run the local search algorithms.
- **README.md**: This document.
- **LICENSE**: License information.
- **requirements.txt**: List of dependencies.

---

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/Eval-LocalSearch.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd Eval-LocalSearch
    ```

3. **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have TensorFlow, NumPy, and other necessary libraries installed.

---

## Usage

### Configuration

Before running the algorithms, configure the settings in `config.py`:

- **Random Seed**: Set `SEED` for reproducibility.
- **Evaluation Metric**: Set `EVALUATION_METRIC` to `'PSNR'` or `'SynFlow'`.
- **Local Search Parameters**: Adjust `LOCAL_SEARCH_CONFIG` with parameters like `MAX_EVALUATIONS`, `TABU_TENURE`, etc.

```python
# config.py

# Set random seeds for reproducibility
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation Metric
EVALUATION_METRIC = 'SynFlow'  # Change to 'PSNR' if needed

# Local Search Configuration
LOCAL_SEARCH_CONFIG = {
    'MAX_EVALUATIONS': 25000,
    'TABU_TENURE': 5,
    'INITIAL_TEMP': 100,
    'COOLING_RATE': 0.95,
    'MUTATION_PROB': 0.1
}

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
```

### Running the Algorithms

Use `main.py` to execute the local search algorithms. Select the algorithm by setting `algorithm_choice`.

```python
# In main.py
algorithm_choice = 'HillClimbing'  # Options: 'HillClimbing', 'TabuSearch', 'SimulatedAnnealing'
```

Run the script:

```bash
python main.py
```

**Note**: Ensure you have implemented your own `decode` function and model builder to integrate your search space.

---

## Algorithms Implemented

### Local Search Methods

#### Hill Climbing

An iterative algorithm that starts with a random solution and makes incremental changes to improve it based on the objective function.

#### Tabu Search

An advanced method that utilizes a tabu list to store recently visited solutions, preventing the algorithm from cycling back and encouraging exploration of new areas.

#### Simulated Annealing

A probabilistic technique that allows the exploration of the search space by accepting worse solutions with a certain probability, which decreases over time.

---

## Research Publications

This project is part of ongoing research in neural architecture search and optimization algorithms.

**Note**: Please add any relevant publications or research articles if available.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix.
3. **Commit your changes** with clear messages.
4. **Push to your fork**.
5. **Submit a pull request**.

Please ensure your code adheres to the project's coding standards and includes proper documentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Quick Links

- [Project Repository](https://github.com/yourusername/Eval-LocalSearch)
- [Issues](https://github.com/yourusername/Eval-LocalSearch/issues)
- [Pull Requests](https://github.com/yourusername/Eval-LocalSearch/pulls)

---

Thank you for your interest in this project!
