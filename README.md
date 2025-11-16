# 🎓 Data Science Thesis: Evolutionary Deep Learning

## Project Overview

This thesis, completed at **Northeastern University** under the supervision of **Professor Rachlin**, focuses on **Optimizing Automated Neural Architecture Search (NAS) using Multi-Objective Evolutionary Algorithms (MOEA).** The goal is to move beyond conventional accuracy-only optimization to develop a framework that balances performance with crucial real-world constraints.

The initial investigation applied an Evo Framework, adapted from Professor Rachlin's code, to evolve feed-forward networks using a Many-Objectives approach on tabular data. However, a comprehensive literature review prompted a significant pivot toward a more **scientifically and societally impactful** focus. The current research is centered on evolving **Recurrent Neural Networks (RNNs)** for **multi-classification of audio data**, specifically tackling the challenge of **high-resource vs. low-resource languages**.


## Methodology

The MOEA framework evolves neural architectures by optimizing for the non-dominated **Pareto optimal solution set** based on **four primary objectives:**
1.  **Macro Model Performance:** Maxmizing accuracy while minimizng type I and II errors
2.  **Model Complexity (Interpretability):** Minimizing complexity to improve model transparency and explainability
3.  **Resource Utilization:** Minimizing computational costs while maximizing computational efficiency
4.  **Algorithmic Fairness:** Addressing algorithmic bias, particularly across diverse language groups


## Value Proposition
By integrating **Algorithmic Fairness** and **Resource Utilization** directly into the evolutionary loop of NAS, this research makes a direct contribution to discussions surrounding **deployable, equitable, and efficient AI systems**, moving the field toward more responsible and robust solutions. Majority of research contributions in evolutionary deep learning have largely focused on using EAs to optimize convolutional neural networks, with limited attention given to recurrent neural networks or generative AI.  