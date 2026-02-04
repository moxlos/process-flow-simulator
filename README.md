# Process Flow Simulator

A discrete-event simulation tool for modeling and optimizing sequential workflows across any industry. Built with Python, SimPy, Streamlit, and Plotly.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SimPy](https://img.shields.io/badge/SimPy-4.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## Overview

This simulation tool allows users to:

- **Configure workers** with specific task capabilities and queue sizes
- **Define products/services** that must pass through a sequence of tasks
- **Set task time distributions** (Normal, Uniform, Exponential) to reflect real-world variability
- **Run simulations** and visualize task assignments with interactive Gantt charts
- **Analyze performance** through comprehensive metrics and analytics dashboards

<img width="2508" height="1222" alt="Screenshot from 2026-02-04 3" src="https://github.com/user-attachments/assets/dc8d1d56-8188-477c-9292-55f409d029de" />
<img width="2508" height="1222" alt="Screenshot from 2026-02-04 2" src="https://github.com/user-attachments/assets/2417eb46-b305-4a59-9473-1919bffff09f" />
<img width="2508" height="1222" alt="Screenshot from 2026-02-04" src="https://github.com/user-attachments/assets/8ccb002c-98d5-4436-8cce-b01c66048e27" />



## Features

### Simulation Engine
- Discrete-event simulation using SimPy
- Worker task queues with configurable capacity
- Product task sequences with dependency ordering
- Configurable time distributions for task completion

### Visualization & Analytics
- **Gantt Chart**: Timeline visualization of task assignments and progress
- **Worker Utilization**: Bar charts showing worker activity rates
- **Task Completion Times**: Average time analysis per task type
- **Product Completion Distribution**: Histogram of completion times
- **Cumulative Progress**: Line plots for task and product completion over time

### Metrics & Reports
- Worker utilization rates
- Average task completion time by worker
- Task frequency distribution
- Product completion times
- Worker idle time analysis
- Queue length statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/moxlos/process-flow-simulator.git
cd process-flow-simulator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

```bash
streamlit run app.py
```

Navigate through the sidebar to:
1. **Configuration**: Set up workers, products, and tasks via UI or JSON file
2. **Simulation**: View the Gantt chart of the simulation run
3. **Logs**: Review detailed simulation logs
4. **Report**: View performance metrics
5. **Analytics**: Explore interactive charts and statistics

### Command Line

```bash
python simulation.py
```

Runs a simulation with default parameters and outputs metrics to console.

## Configuration

### JSON Configuration Format

Save/load configurations in `data/configurations.json`:

```json
{
  "tasks": {
    "Task Name": {
      "distribution": "normal_dist",
      "parameters": [10.0, 2.0]
    }
  },
  "workers": [
    {
      "id": "01",
      "tasks": ["Task Name"],
      "max_queue_size": 2
    }
  ],
  "products": [
    {
      "id": "01",
      "tasks": ["Task Name"]
    }
  ]
}
```

### Supported Distributions

| Distribution | Parameters | Description |
|-------------|------------|-------------|
| `normal_dist` | `[mean, stddev]` | Normal distribution |
| `uniform_dist` | `[min, max]` | Uniform distribution |
| `exponential_dist` | `[scale]` | Exponential distribution |

## Project Structure

```
process-flow-simulator/
├── app.py                 # Streamlit web application
├── simulation.py          # Core simulation engine
├── utils.py               # Visualization and metrics utilities
├── logging_module.py      # Shared logging data structures
├── requirements.txt       # Python dependencies
├── data/
│   ├── configurations.json    # User configurations
│   └── examples/              # Example configurations
│       ├── 01_small_team.json
│       ├── 02_large_team.json
│       ├── 03_bottleneck_scenario.json
│       ├── 04_healthcare_patient_flow.json
│       └── 05_manufacturing_assembly.json
└── README.md
```

## Technical Details

### Core Classes

- **Task**: Represents a task with a name and time distribution
- **Worker**: SimPy-based worker with task queue and capability list
- **Product**: Item that flows through a sequence of tasks

### Simulation Logic

1. Products are assigned to available workers based on task requirements
2. Workers process tasks from their queue sequentially
3. Each task duration is sampled from the configured distribution
4. Products move to the next task in their sequence upon completion
5. Simulation ends when all products complete their task sequences

## Industry Applications

Discrete-event simulation is widely used for workflow optimization in:

### Healthcare
- **Patient flow analysis** in emergency departments and clinics
- Scheduling operating rooms and medical staff
- Reducing wait times and improving throughput
- Capacity planning for hospitals

### Manufacturing
- **Assembly line balancing** and throughput optimization
- Identifying production bottlenecks
- Resource allocation and shift planning
- Quality control process design

## Example Configurations

Pre-built example configurations are available in `data/examples/`:

| File | Scenario | Description |
|------|----------|-------------|
| `01_small_team.json` | Small Team | 3 loan officers processing 5 applications |
| `02_large_team.json` | Large Team | 10 insurance adjusters handling 20 claims |
| `03_bottleneck_scenario.json` | Bottleneck Analysis | Permit office with inspection bottleneck |
| `04_healthcare_patient_flow.json` | Healthcare | Emergency department patient flow simulation |
| `05_manufacturing_assembly.json` | Manufacturing | Electronics assembly line with 6 stations |

### Using Examples

1. In the Streamlit app, select **"From File"** configuration method
2. Upload any example JSON file from `data/examples/`
3. Click **"Run Simulation"** to see the results

### What to Analyze

- **Small Team**: Good for understanding basic workflow dynamics
- **Large Team**: Observe how specialization affects utilization
- **Bottleneck**: See how a single constrained resource impacts the entire system
- **Healthcare**: Analyze patient wait times and staff utilization
- **Manufacturing**: Study line balancing and station throughput

## License

### GNU General Public License v3.0
