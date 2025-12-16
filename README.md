# Robot Localization with Hidden Markov Models

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive implementation of a Hidden Markov Model (HMM) for robot localization in a grid-world environment. This project demonstrates probabilistic filtering and Viterbi decoding to estimate a robot's position given noisy sensor observations.

## 📋 Overview

This project implements:

- **State Space**: 2D grid world with position (x, y) and orientation (N, E, S, W)
- **Transition Model**: Action-dependent (Forward, Turn Left, Turn Right, Stay) with stochastic noise
- **Observation Model**: 
  - Beacon sensor: Binary detection of nearby beacons with distance-dependent probability
  - Wall distance sensor: Noisy distance measurement to nearest wall ahead
- **Inference Algorithms**:
  - **Forward Filtering**: Real-time belief state computation
  - **Viterbi Decoding**: Most likely state sequence recovery

## 🎯 Key Features

✅ Realistic probabilistic models with multiple sensor types  
✅ Efficient transition and observation probability matrices  
✅ Interactive visualization with real-time belief updates  
✅ Comprehensive trajectory simulation  
✅ Support for custom grid dimensions and sensor parameters  

## 📁 Project Structure

```
.
├── main.py           # Core HMM implementation (GridWorldHMM class)
├── display.py        # Interactive visualization
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VictimPickle/robot-localization-hmm.git
cd robot-localization-hmm

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Run the interactive visualization
python display.py

# Press SPACE to advance through time steps
```

## 📊 How It Works

### State Representation

Each state encodes:
- **Position**: (x, y) in a grid of dimensions W × H
- **Orientation**: One of {N, E, S, W} (North, East, South, West)
- **Total States**: W × H × 4

### Transition Model

For action `F` (Forward):
- 70% probability: move forward
- 12.5% probability: move left-forward  
- 12.5% probability: move right-forward
- 5% probability: stay in place

### Observation Model

**Beacon Sensor:**
- P(see | distance = 0) = 0.85
- P(see | distance = 1) = 0.40
- P(see | distance ≥ 2) = 0.10

**Wall Sensor:**
- Observes distance class {0, 1, 2} to nearest wall
- Probabilities depend on true distance ahead

### Forward Filtering Algorithm

```
Bel(S_t) = η · P(O_t | S_t) · Σ_s P(S_t | S_{t-1}, A_t) · Bel(S_{t-1})
```

Where:
- `Bel(S_t)`: Probability distribution over states at time t
- `P(O_t | S_t)`: Observation model
- `P(S_t | S_{t-1}, A_t)`: Transition model
- `η`: Normalization factor

### Viterbi Decoding

Finds the most likely state sequence given observations:
```
δ_t(s) = max_s' [δ_{t-1}(s') · P(S_t=s | S_{t-1}=s', A_t) · P(O_t | S_t=s)]
```

Backtracking recovers the optimal path S₀* → S₁* → ... → S_T*

## 💻 Core Classes and Methods

### GridWorldHMM

**Initialization:**
```python
hmm = GridWorldHMM(
    width=4,              # Grid width
    height=5,             # Grid height
    p_forward=0.7,        # Forward movement success probability
    p_side=0.25,          # Side movement probability
    p_forward_stay=0.05,  # Stay probability
    seed=42               # Random seed
)
```

**Key Methods:**

| Method | Purpose |
|--------|----------|
| `simulate_trajectory(T_steps)` | Generate synthetic robot trajectory |
| `forward_filter(actions, observations)` | Compute belief over time (filtering) |
| `viterbi(actions, observations)` | Find most likely state sequence |
| `state_to_tuple(idx)` | Convert state index to (x, y, orientation) |
| `observation_prob(state, obs)` | Compute P(observation \| state) |

## 📈 Example Output

```
=== Simulated trajectory ===
t= 0 STATE (true) = (x=1, y=2, dir=N) (no obs)
t= 1 a=F STATE (true) = (x=1, y=1, dir=N), obs=(True, 2)
t= 2 a=L STATE (true) = (x=1, y=1, dir=W), obs=(False, 1)

=== Filtering: argmax state at each t ===
t= 0 MAP state = (x=1, y=2, dir=N), prob=0.050
t= 1 MAP state = (x=1, y=1, dir=N), prob=0.245
t= 2 MAP state = (x=1, y=1, dir=W), prob=0.312

=== Viterbi most likely path ===
log P(path*, obs | model) = -145.234
```

## 🔧 Customization

### Modify Grid Dimensions
```python
hmm = GridWorldHMM(width=10, height=8)
```

### Custom Beacon Positions
```python
hmm = GridWorldHMM(
    beacon_positions=[(0, 0), (5, 4), (3, 7)]
)
```

### Adjust Sensor Reliability
```python
hmm = GridWorldHMM(
    p_forward=0.8,      # More reliable forward movement
    p_turn_success=0.95  # More reliable turns
)
```

## 🎓 Educational Value

This project illustrates:

1. **Probabilistic Graphical Models**: HMM structure and inference
2. **Bayesian Filtering**: Real-time state estimation
3. **Dynamic Programming**: Viterbi algorithm for sequence decoding
4. **Numerical Stability**: Handling underflow with log-space computation
5. **Sensor Fusion**: Combining multiple noisy sensors

## 📚 References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

## ⚠️ Important Notes

**Numerical Stability:**  
When running long trajectories (>500 steps), the forward filtering algorithm may encounter numerical underflow. This implementation uses normalization at each step to maintain numerical stability. For production systems, consider log-domain computation.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Log-domain computation for extreme underflow prevention
- [ ] Parallel trajectory simulation
- [ ] Extended Kalman Filter comparison
- [ ] Real sensor data integration

## 📄 License

MIT License © 2025

## 💬 Questions?

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Last Updated**: December 2025  
**Python Version**: 3.8+
