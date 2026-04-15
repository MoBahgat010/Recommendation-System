# xDeepFM Recommendation Engine

This repository provides a highly robust recommendation system implementing the **xDeepFM** architecture. It is designed to capture deep feature representations while retaining the ability to memorize sparse features. By explicitly defining the order of feature correlations, this engine achieves high customizability and maintainability.

## Engine Architecture

The system consists of two primary components:

### 1. Encoder
A multimodal embedding model (*jinaai/jina-clip-v2*) responsible for encoding text and images into a unified semantic feature space.

### 2. Ranker (xDeepFM)
The core recommendation engine that handles complex feature relations and generates suggestions. The ranker model is composed of three main networks:
* **Linear Layer:** Handles the memorization of sparse features.
* **Deep Layers:** Generalizes unobserved feature interactions.
* **Compressed Interaction Network (CIN):** Captures high-order feature interactions explicitly at the vector level, controlled by the interaction matrix size ($m$).

## Future Improvements
- Encode images with text to get a more representative embedding for user and item.
