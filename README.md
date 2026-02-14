# PAQ in Color Vision

[![ICML 2024](https://img.shields.io/badge/ICML-2024-blue)](https://icml.cc/virtual/2024/38215)
[![Allerton 2024](https://img.shields.io/badge/Allerton-2024-green)](https://ieeexplore.ieee.org/document/10735273/)
[![arXiv](https://img.shields.io/badge/arXiv-2309.04626-b31b1b.svg)](https://arxiv.org/abs/2309.04626)

**Learning the Eye of the Beholder: Statistical Modeling and Estimation for Personalized Color Perception**

*Xuanzhou Chen, Austin Xu, Jingyan Wang, and Ashwin Pananjady*

Georgia Institute of Technology

---

## Overview

Traditional approaches to color perception classify individuals as either "color-normal" or "color-blind" with a few distinct categories. However, this binary classification fails to capture the rich spectrum of individual variation in color perception. This project introduces a novel framework for understanding and quantifying personalized color perception at the individual level.

### Key Contributions

- **Unified Color Perception Model**: A mathematical framework that unifies theories for both color-normal and color-deficient vision, revealing low-dimensional structure in how individuals distinguish colors
- **Perceptual Adjustment Queries (PAQs)**: An innovative data collection paradigm that is both highly informative and cognitively lightweight, requiring minimal user effort
- **Statistical Guarantees**: Rigorous theoretical analysis with sample complexity bounds for learning individual color perception profiles
- **Efficient Implementation**: Practical algorithms that learn personalized color distinguishability from a small number of user interactions

### What are Perceptual Adjustment Queries?

Unlike traditional comparison-based similarity queries ("Which of these two colors is more similar to this reference?"), PAQs use an inverted measurement scheme. Users are presented with a color and asked to adjust it until it matches their perception of a target. This approach:

- Combines advantages of both cardinal (quantitative) and ordinal (ranking) queries
- Reduces cognitive load on participants
- Provides richer information per query
- Enables faster convergence to accurate models

## Publications

This work has been presented and published at:

1. **ICML 2024 Workshop** (Humans, Algorithmic Decision-Making and Society)
   - [Virtual Conference Page](https://icml.cc/virtual/2024/38215)
   - [OpenReview](https://openreview.net/forum?id=rXxFQYZetF)

2. **Allerton 2024** (60th Allerton Conference on Communication, Control, and Computing)
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/10735273/)

3. **Related Work**: [Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning](https://arxiv.org/abs/2309.04626) (NeurIPS 2023)

## Repository Structure

```
paq-in-color-vision/
├── simulation/           # Core simulation and statistical analysis code
│   └── ...              # Python scripts for PAQ algorithms and experiments
├── ui_local/            # Local web interface for color perception tests
│   └── ...              # HTML/JavaScript for interactive user studies
├── ui-archive/          # Archived UI versions and experimental interfaces
├── model_rejection.json # Rejection sampling data for model validation
├── figures.pptx        # Supplementary figures (ellipses, copunctal points)
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Modern web browser (for UI components)
- Required Python packages:

```bash
pip install numpy scipy matplotlib scikit-learn
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xchen793/paq-in-color-vision.git
cd paq-in-color-vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If available
```

3. For local UI testing:
```bash
cd ui_local
# Open index.html in your browser or run a local server:
python -m http.server 8000
# Navigate to http://localhost:8000
```

## Usage

### Running Simulations

The `simulation/` directory contains Python scripts for:
- Generating synthetic color perception data
- Running PAQ-based estimation algorithms
- Comparing PAQ performance vs traditional similarity queries
- Statistical testing and validation

Example usage:
```python
# Example workflow (adjust based on actual scripts)
from simulation import paq_estimator

# Initialize color space and user model
user_model = create_user_model(rank=2)  # Low-rank color perception

# Collect PAQ measurements
queries = generate_paq_queries(n_queries=50)
responses = user_model.respond(queries)

# Estimate color perception matrix
estimated_metric = paq_estimator.fit(queries, responses)

# Evaluate performance
error = compute_estimation_error(estimated_metric, user_model.true_metric)
```

### Conducting User Studies

The interactive web interface allows you to:
1. Collect PAQ responses from human participants
2. Build personalized color perception profiles
3. Visualize individual color distinguishability

To run a user study:
1. Navigate to `ui_local/`
2. Launch the web interface
3. Participants adjust colors using intuitive sliders
4. Results are saved for downstream analysis

### Data Format

PAQ response datasets include:
- Query parameters (reference color, target color)
- User adjustments (RGB values)
- Response timestamps
- Participant metadata (anonymized)

## Methodology

### Color Perception Model

We model individual color perception using a low-rank Mahalanobis distance in color space:

```
d_M(c₁, c₂) = √[(c₁ - c₂)ᵀ M (c₁ - c₂)]
```

where M is a positive semidefinite matrix encoding how the user distinguishes colors. The low-rank structure captures that color perception is governed by a small number of underlying dimensions (e.g., loss of one cone type in dichromacy).

### Two-Stage Estimator

1. **Stage 1**: Initial spectral method to obtain coarse estimate
2. **Stage 2**: Refinement via local optimization
3. **Theoretical guarantees**: Sample complexity scales with intrinsic dimensionality

## Results

Our experiments demonstrate:

- **Sample Efficiency**: PAQs require 2-3× fewer queries than similarity comparisons to achieve the same accuracy
- **Cognitive Load**: Participants report PAQs as easier and more intuitive
- **Individual Variation**: Significant heterogeneity in color perception even among "color-normal" individuals
- **Clinical Relevance**: Potential applications in precise diagnosis and assistive technology

See `figures.pptx` for visualizations including:
- Color discrimination ellipses
- Copunctal point analysis
- Comparison of estimation methods

## Applications

This framework enables:

- **Personalized color interfaces**: Adapt displays to individual perception
- **Assistive technology**: Better tools for color-deficient individuals  
- **Clinical diagnosis**: Fine-grained characterization beyond categorical labels
- **Vision science research**: Understanding individual differences in color perception
- **Design and accessibility**: Create universally perceivable color schemes

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{chen2024learning,
  title={Learning the eye of the beholder: Statistical modeling and estimation for personalized color perception},
  author={Chen, Xuanzhou and Xu, Austin and Wang, Jingyan and Pananjady, Ashwin},
  booktitle={ICML Workshop on Humans, Algorithmic Decision-Making and Society},
  year={2024}
}
```

For the foundational PAQ methodology:
```bibtex
@inproceedings{xu2023perceptual,
  title={Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning},
  author={Xu, Austin and McRae, Andrew and Wang, Jingyan and Davenport, Mark A and Pananjady, Ashwin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or request features via GitHub Issues
- Submit pull requests with improvements
- Share your applications of the PAQ framework

## Contact

For questions, comments, or collaborations:

**Xuanzhou Chen**  
Ph.D. Student, Machine Learning  
Georgia Institute of Technology  
Email: xchen920@gatech.edu  
Website: [xchen793.github.io](https://xchen793.github.io/)

**Principal Investigator: Ashwin Pananjady**  
Assistant Professor  
Georgia Institute of Technology  
[Faculty Page](https://sites.gatech.edu/ashwin-pananjady/)

## License

[Specify license - e.g., MIT, Apache 2.0, GPL, etc.]

## Acknowledgments

This work was supported by [funding sources, if applicable]. We thank all participants in our user studies for their valuable contributions.

---

## Related Resources

- [OpenReview Discussion](https://openreview.net/forum?id=rXxFQYZetF)
- [ICML 2024 Virtual Poster](https://icml.cc/virtual/2024/38215)
- [NeurIPS 2023 PAQ Paper](https://arxiv.org/abs/2309.04626)
- [Georgia Tech ML@GT Group](https://ml.gatech.edu/)

## Updates

- **October 2024**: Presented at 60th Allerton Conference
- **July 2024**: Accepted to ICML 2024 Workshop
- **2024**: Initial repository release with simulation code and user study interface

---

**Keywords**: color perception, perceptual adjustment queries, metric learning, low-rank estimation, personalized vision, color deficiency, statistical learning theory, human-computer interaction
