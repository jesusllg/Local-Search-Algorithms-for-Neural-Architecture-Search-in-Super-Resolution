# Evaluation and Local Search Algorithms for Neural Architecture Search in Super Resolution

## Introduction

**Eval-LocalSearch** provides a suite of evaluation functions and local search algorithms designed for Neural Architecture Search (NAS). This repository is flexible and can be integrated with any search space as long as it uses a binary encoding.

The primary goal is to refine and optimize neural network architectures by exploring the search space locally, balancing multiple objectives:

- **Performance**: Measured by PSNR or SynFlow.
- **Model Complexity**: Number of parameters.
- **Computational Cost**: Measured in FLOPs.

---

## Features

- **Flexible Integration**: Compatible with any binary-encoded search space.
- **Local Search Algorithms**: Implements Hill Climbing, Tabu Search, and Simulated Annealing.
- **Multi-Objective Evaluation**: Supports PSNR, SynFlow, parameter count, and FLOPs.
- **Customizable Metrics**: Easily integrate your own evaluation metrics.
- **Modular Design**: Organized code structure for easy understanding and modification.
- **Extensive Documentation**: Thoroughly commented code and comprehensive README.

---

## Performance Comparison

As part of the optimization process, the local search algorithms aim to find architectures that are locally optimal based on the defined objectives.

**Note**: Specific performance metrics will depend on your search space, datasets, and configurations. It's recommended to log and analyze results after running the algorithms.

---

## How It Works

### Integration with Search Spaces

- **Search Space Agnostic**: Designed to be independent of any specific search space.
- **Encoding**: Requires architectures to be represented using binary encoding.
- **Decoding Function**: Users must provide their own decoding function to convert the binary code into a model architecture.

### Evaluation Metrics

- **PSNR**: Measures image reconstruction quality.
- **SynFlow**: Proxy metric for evaluating architectures without full training.
- **FLOPs and Parameters**: Assess computational cost and model complexity.
- **Custom Metrics**: Integrate additional metrics as needed.

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
├── model_builder.py
├── main.py
├── README.md
├── LICENSE
└── requirements.txt
```

- **config.py**: Configuration settings, including evaluation metrics and local search parameters.
- **evaluation.py**: Evaluation functions for PSNR, SynFlow, FLOPs, and parameter counting.
- **local_search.py**: Implementation of local search algorithms.
- **utils.py**: Utility functions, such as dominance checks.
- **model_builder.py**: Functions to build models based on decoded genomes.
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

    Ensure you have TensorFlow and NumPy installed. If you plan to use additional libraries, add them to `requirements.txt`.

---

## Usage

### Configuration

Before running the algorithms, configure the settings in `config.py`:

- **Random Seed**: Set `SEED` for reproducibility.
- **Evaluation Metric**: Set `EVALUATION_METRIC` to `'PSNR'` or `'SynFlow'`.
- **Local Search Parameters**: Adjust `LOCAL_SEARCH_CONFIG` with parameters like `MAX_EVALUATIONS`, `TABU_TENURE`, etc.
- **Dataset Configuration**: Replace the placeholder datasets with your actual training and validation datasets.

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

**Note**: Ensure you have implemented your own `decode` and `get_model` functions to integrate your search space.

#### Implementing `decode` and `get_model`

To use Eval-LocalSearch with your specific search space, you need to implement the `encoding/decoding` and `get_model` functions in `main.py`. Here's how you can do it:

1. **Implement `decode` Function:**

    The `decode` function converts a binary genome into a structured representation of your neural network architecture.

    ```python
    def decode(genome):
        """
        Decode the binary genome into a neural network architecture.
        Users must implement this function according to their encoding scheme.
        
        Args:
            genome (list or ndarray): Binary encoding of the architecture.
        
        Returns:
            dict: Decoded architecture parameters.
        """
        # Example placeholder implementation
        architecture = {}
        # Replace with your actual decoding logic
        # Example:
        # architecture['layers'] = []
        # for i in range(0, len(genome), 8):
        #     layer_type = genome[i]
        #     if layer_type == 0:
        #         architecture['layers'].append({'type': 'Conv', 'filters': genome[i+1:i+5], ...})
        #     elif layer_type == 1:
        #         architecture['layers'].append({'type': 'Dense', 'units': genome[i+1:i+5], ...})
        return architecture
    ```

2. **Implement `get_model` Function:**

    The `get_model` function builds and returns a TensorFlow/Keras model based on the decoded architecture.

    ```python
    def get_model(decoded_genome):
        """
        Build and return a TensorFlow/Keras model based on the decoded genome.
        Users must implement this function according to their architecture specifications.
        
        Args:
            decoded_genome (dict): Decoded architecture parameters.
        
        Returns:
            tf.keras.Model: The constructed Keras model.
        """
        # Example placeholder implementation
        model = tf.keras.Sequential()
        
        # Example: Add layers based on decoded_genome
        for layer_info in decoded_genome.get('layers', []):
            if layer_info['type'] == 'Conv':
                model.add(tf.keras.layers.Conv2D(
                    filters=layer_info['filters'],
                    kernel_size=layer_info['kernel_size'],
                    activation=layer_info.get('activation', 'relu'),
                    input_shape=layer_info.get('input_shape')
                ))
            elif layer_info['type'] == 'Dense':
                model.add(tf.keras.layers.Dense(
                    units=layer_info['units'],
                    activation=layer_info.get('activation', 'relu')
                ))
            elif layer_info['type'] == 'Pooling':
                model.add(tf.keras.layers.MaxPooling2D(pool_size=layer_info['pool_size']))
            # Add more layer types as needed
        
        # Example: Add output layer
        model.add(tf.keras.layers.Dense(3, activation='sigmoid'))  # Adjust based on your task

        return model
    ```

Ensure these functions accurately reflect your specific neural architecture search space.

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

Thank you for your interest in Eval-LocalSearch!
```
