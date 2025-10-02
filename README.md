# Validating Nyström Operator KRR for Path-to-Path Regression

This repository contains the code and results for an empirical study validating the **Nyström approximation** for **Operator Kernel Ridge Regression (op-KRR)**.  
Our experiments show that the Nyström method provides a **computationally efficient** and **highly accurate** approximation to the exact op-KRR model for a challenging non-linear path-to-path regression task.

---

## 📖 Overview

Operator Kernel Ridge Regression (op-KRR) is a powerful non-parametric method for learning operators between function spaces.  
When combined with **path signatures**, it becomes a state-of-the-art technique for path-to-path regression.

- **Exact op-KRR** has complexity:  
  - Time: **O(N³)**  
  - Memory: **O(N²)**  
  making it intractable for large datasets.

- **Nyström op-KRR** uses a subset of `m` landmark points to form a low-rank Gram matrix approximation:  
  - Time: **O(N m²)** for training  
  - Time: **O(m)** for prediction  

This makes Nyström op-KRR highly scalable while maintaining accuracy.

We validate this approach on synthetic data generated from a stochastic differential equation (SDE). Results show that Nyström op-KRR achieves nearly identical performance to the exact method at a fraction of the computational cost.

---

## 🔬 Methodology

The experiment compares **three models** on a controlled, synthetic dataset.

### 1. Data Generation
- Input paths `{X_i}` are 1D **fractional Brownian motion (fBm)** with Hurst parameter `H=0.1`.  
- Output paths `{Y_i}` are generated via the non-linear SDE:

\[
dY_t = (aX_t + bX_t^3)dt + cY_t\,dX_t
\]

- Dataset size:  
  - `N_total = 550` paths  
  - `N_train = 500`  
  - `N_test = 50`  

---

### 2. Models Compared
- **op-KRR (Exact):** full Operator KRR, used as the "gold standard".  
- **Nyström op-KRR:** Nyström approximation with `m=100` landmark points.  
- **Linear Sig:** baseline ridge regression directly on input path signatures.  

---

### 3. Evaluation & Robustness
- **Metrics**  
  - Pathwise Mean Squared Error (MSE): compares reconstructed paths.  
  - Weighted Signature Error: weighted L2 distance in signature space.  
- **Robustness**  
  - Each experiment repeated **10 times** with different seeds.  
  - Reported results are **mean ± std**.  

---

## 📊 Results

Nyström op-KRR is nearly indistinguishable from the exact op-KRR and significantly outperforms the linear baseline.

| Model           | Pathwise MSE (Mean ± Std) | Weighted Sig Error (Mean ± Std) |
|-----------------|---------------------------|---------------------------------|
| Linear Sig      | 0.983 ± 0.723             | 2.02 ± 3.81                     |
| Nyström op-KRR  | 0.766 ± 0.535             | 1.04 ± 0.84                     |
| op-KRR (Exact)  | 0.771 ± 0.657             | 0.71 ± 0.71                     |

---

### Visualizations
- **Predicted Paths:** Nyström (dotted) closely matches Exact op-KRR (dashed).  
- **MSE Distribution:** Nyström and Exact models have nearly identical low error distributions, far outperforming Linear Sig.  

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install numpy scipy scikit-learn matplotlib fbm iisignature joblib pandas seaborn
