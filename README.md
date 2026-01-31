# SMART-VISION-FUSION: Multimodal Vulnerability Detection for Smart Contracts

## Overview  
This repository contains the official implementation of **SMART-VISION-FUSION**, a novel dual-branch multimodal framework for detecting vulnerabilities in smart contracts. The method integrates deep semantic representations from large language models (CodeQwen-1.5-7B) with structural dependencies captured via a Heterogeneous Graph Transformer (HGT). A cross-attention fusion mechanism enables fine-grained interaction between the two modalities, significantly improving detection accuracy, generalization, and adversarial robustness compared to existing approaches.

## Key Features  
- **Dual-Branch Architecture**: Processes both semantic (LLM-based) and structural (graph-based) representations in parallel.
- **Cross-Attention Fusion**: Enables dynamic, fine-grained alignment between semantic tokens and structural graph nodes.
- **Heterogeneous Contract Graph (HCG)**: Captures Solidity-specific constructs, control flow, data dependencies, and inheritance hierarchies.
- **Pretrained LLM Integration**: Leverages CodeQwen-1.5-7B for rich contextual code understanding.
- **Multi-Vulnerability Detection**: Supports nine vulnerability types including reentrancy, arithmetic flaws, access control issues, front-running, and more.
- **Rigorous Evaluation**: Benchmarked on DappSCAN and SmartBugs-Curated with comprehensive cross-dataset and adversarial robustness testing.

## Main Contributions  
1. **Deep Multimodal Fusion**: Proposes a shift from shallow feature concatenation to interactive cross-attention fusion.
2. **Unidirectional Cross-Attention**: Structural embeddings selectively attend to semantic cues, reducing noise and improving alignment.
3. **State-of-the-Art Performance**: Demonstrates significant improvements, particularly on arithmetic vulnerabilities (+22.1% F1 gain).
4. **Enhanced Robustness**: Shows stronger generalization and adversarial resistance compared to unimodal and existing hybrid baselines.

## Results Highlights  
- **In-Dataset Performance**: Achieves **0.8540 macro-F1** on DappSCAN, outperforming unimodal models by 15–40%.
- **Cross-Dataset Generalization**: **19.9% relative improvement** over the best single-modality baseline on SmartBugs-Curated.
- **Adversarial Robustness**: **27.51% Attack Success Rate (ASR)** under LLM-based rewriting vs. 35.29% for HGT-only model.
- **Superior to Hybrid SOTA**: Outperforms DA-GNN and BSGVD by **7.1%** on average F1 score.

## Methodology  
The framework consists of three core stages:

1. **Structural Encoding**:  
   - Constructs a Heterogeneous Contract Graph (HCG) from Solidity source code.
   - Encodes the graph using a Heterogeneous Graph Transformer (HGT) with relation-aware attention.

2. **Semantic Embedding**:  
   - Extracts contextual embeddings using the frozen CodeQwen-1.5-7B model.
   - Applies mean pooling over token representations and projects them into the HGT latent space.

3. **Cross-Attention Fusion**:  
   - Uses structural embeddings as queries to attend to semantic keys and values.
   - Produces a unified representation for final vulnerability classification via a two-layer MLP.

## Limitations & Future Directions  
- **Computational Overhead**: LLM inference introduces significant cost; optimization via distillation or compression is needed.
- **Generalization Gap**: Performance drop in cross-dataset settings indicates need for domain adaptation techniques.
- **Vulnerability-Specific Performance**: Timestamp-related patterns remain challenging for the current fusion design.
- **Future Work**: May explore gated/hierarchical attention, model compression, dynamic execution traces, and certified defenses.

## Citation  
If you use this code or findings in your research, please cite the original paper:

```bibtex
@article{giang2025smartvisionfusion,
  title={Multimodal Smart Contract Vulnerability Detection via Semantic-Structural Cross-Attention},
  author={Le Huynh Giang and Chu Nguyen Hoang Phuong and Triet Huynh Minh Le and M. Ali Babar and Van-Hau Pham and Phan The Duy},
  journal={Journal of Smart Contract Vulnerability Detection},
  year={2025}
}
```
## License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Contact  
For questions or collaborations, please:  
- **Open an issue** in this repository, or  
- **Contact the authors** via the institutional emails provided in the original paper.
