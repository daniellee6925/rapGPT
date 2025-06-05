# rapGPT 1.0: A Transformer-Based Rap Lyric Generator

## Overview

**rapGPT** is a decoder-only Transformer language model for generating rap lyrics in the style of Eminem. The model is trained on a curated dataset of Eminem's lyrics sourced from Kaggle and leverages autoregressive language modeling to generate one token at a time.

This project aims to explore the fundamentals of a transformer architectures and pretraining large language models. 

**Note**: This version of the model produces limited results. For improved performance and a deployed interface, please check out [rapGPT 2.0](https://github.com/daniellee6925/rapGPT2.0)

---

## Lyric Data
You can access the Emeinem lyrics dataset from [Kaggle](https://www.kaggle.com/datasets/aditya2803/eminem-lyrics/data)

Aditya2803. (2021). Eminem Lyrics Dataset [Dataset]. Kaggle.

## Model Architecture

The architecture follows a decoder-only Transformer design. Key components include:

- **Stacked Decoder Blocks** with:
  - Multi-head self-attention
  - Feedforward neural networks
- **Autoregressive Generation**: Predicts the next token given prior context
- **Positional Embeddings**: Encodes token order in sequences

---

## Training Setup

The model was trained using a custom training loop with the following hyperparameters:

| Hyperparameter        | Value             |
|-----------------------|-------------------|
| Batch Size            | 16                |
| Block Size            | 512               |
| Optimizer             | AdamW             |
| Learning Rate         | 3e-4              |
| Vocabulary Size       | 30,000            |
| Dropout               | 0.2               |
| Embedding Dimension   | 512               |
| Attention Heads       | 8                 |
| Transformer Layers    | 8                 |

---

## Evaluation

- Computes average **training loss** and **validation loss**
- Helps monitor overfitting and model progress

---

## Output Examples

Even after sufficient training, the model was not able to produce understandable lyrics

---

## Future Work

- Expand dataset to include other artists or genres
- Build a better model for lyric generation 
- Add rhyme density or syllable flow evaluation metrics
- Deploy via a web interface
- please refer to the newer version [rapGPT 2.0](https://github.com/daniellee6925/rapGPT2.0)

---

## License

This project is licensed under the [MIT License](https://opensource.org/license/MIT).

Please refer to `LICENSE`

