# Homework 4 – Unsupervised Learning: Theoretical Answers
**Author:** Claudio Rocca

---

## Exercise 1: SVD of a Matrix and Verification

**Experimental results ($A \in \mathbb{R}^{10 \times 6}$, random Gaussian matrix):**

```
Matrix dimensions: A ∈ ℝ¹⁰ˣ⁶  |  U ∈ ℝ¹⁰ˣ¹⁰  |  Σ ∈ ℝ¹⁰ˣ⁶  |  Vᵀ ∈ ℝ⁶ˣ⁶
Singular values:   [4.5327, 3.7391, 2.7217, 1.8981, 1.5907, 1.4378]
Reconstruction Frobenius error: ||A - U Σ Vᵀ||_F = 6.70e-15
Numerical rank: 6  |  Non-zero singular values: 6
```

---

### Deep Dive: What do $U$, $\Sigma$, and $V$ Represent?

The **Singular Value Decomposition (SVD)** is the fundamental factorization of linear algebra. For any real rectangular matrix $A \in \mathbb{R}^{m \times n}$ of rank $r \leq \min(m,n)$, the full SVD is:
$$A = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{m \times m}$ is an **orthogonal matrix** ($U^T U = I_m, U U^T = I_m$).
- $\Sigma \in \mathbb{R}^{m \times n}$ is a **rectangular diagonal matrix** containing non-negative singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_{\min(m,n)} \geq 0$ on its main diagonal.
- $V \in \mathbb{R}^{n \times n}$ is an **orthogonal matrix** ($V^T V = I_n, V V^T = I_n$).

```
                    ┌                     ┐
                    │ σ₁                  │
┌                 ┐ │    σ₂               │ ┌                     ┐
│                 │ │       ⋱             │ │    — v₁ᵀ —          │
│  |   |     |    │ │          σᵣ         │ │    — v₂ᵀ —          │
│  u₁  u₂ ⋯  uₘ   │ │             0       │ │       ⋮             │
│  |   |     |    │ │                ⋱    │ │    — vₙᵀ —          │
│                 │ │                   0 │ └                     ┘
└                 ┘ └                     ┘         Vᵀ ∈ ℝⁿˣⁿ
    U ∈ ℝᵐˣᵐ              Σ ∈ ℝᵐˣⁿ
```

In reduced (thin / compact) form, keeping only the $r$ non-zero singular values:
$$A = U_r \Sigma_r V_r^T = \sum_{i=1}^r \sigma_i u_i v_i^T$$
where $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r = \operatorname{diag}(\sigma_1, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$, and $V_r \in \mathbb{R}^{n \times r}$.

---

#### 1. What does $V$ represent? (Right Singular Vectors)

- **Algebraic definition:** The columns of $V$, denoted $v_1, v_2, \dots, v_n \in \mathbb{R}^n$, are called the **right singular vectors** of $A$. They are the orthonormal eigenvectors of the symmetric $n \times n$ Gram matrix $A^T A$:
  $$A^T A = (V \Sigma^T U^T)(U \Sigma V^T) = V (\Sigma^T \Sigma) V^T = V \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2) V^T$$
  $$(A^T A) v_i = \sigma_i^2 v_i, \qquad i = 1, \dots, n$$

- **Subspace decomposition:**
  - The first $r$ columns $\{v_1, \dots, v_r\}$ form an orthonormal basis for the **row space** of $A$: $\operatorname{Row}(A) = \operatorname{range}(A^T) \subseteq \mathbb{R}^n$.
  - The remaining $n - r$ columns $\{v_{r+1}, \dots, v_n\}$ form an orthonormal basis for the **nullspace (kernel)** of $A$: $\ker(A) = \{x \in \mathbb{R}^n : A x = 0\}$.

- **Geometric meaning (Domain space $\mathbb{R}^n$):**
  $v_i$ represent the **input principal directions**. They are mutually orthogonal directions in the domain $\mathbb{R}^n$ that the transformation $A$ maps directly onto orthogonal directions in the codomain $\mathbb{R}^m$ without any cross-coupling (cross-talk).

---

#### 2. What does $U$ represent? (Left Singular Vectors)

- **Algebraic definition:** The columns of $U$, denoted $u_1, u_2, \dots, u_m \in \mathbb{R}^m$, are called the **left singular vectors** of $A$. They are the orthonormal eigenvectors of the symmetric $m \times m$ outer Gram matrix $A A^T$:
  $$A A^T = (U \Sigma V^T)(V \Sigma^T U^T) = U (\Sigma \Sigma^T) U^T = U \operatorname{diag}(\sigma_1^2, \dots, \sigma_m^2) U^T$$
  $$(A A^T) u_i = \sigma_i^2 u_i, \qquad i = 1, \dots, m$$

- **Subspace decomposition:**
  - The first $r$ columns $\{u_1, \dots, u_r\}$ form an orthonormal basis for the **column space (range / image)** of $A$: $\operatorname{Col}(A) = \operatorname{range}(A) \subseteq \mathbb{R}^m$.
  - The remaining $m - r$ columns $\{u_{r+1}, \dots, u_m\}$ form an orthonormal basis for the **left nullspace (orthogonal complement of range)**: $\ker(A^T) = \operatorname{range}(A)^\perp \subseteq \mathbb{R}^m$.

- **Geometric meaning (Codomain space $\mathbb{R}^m$):**
  $u_i$ represent the **output principal directions**. They are the normalized axis directions of the hyperellipsoid in $\mathbb{R}^m$ obtained by mapping the unit sphere of $\mathbb{R}^n$ through $A$.

---

#### 3. The Fundamental Bridge: $A v_i = \sigma_i u_i$

Multiplying $A = U \Sigma V^T$ on the right by $V$ yields:
$$A V = U \Sigma \quad \iff \quad A v_i = \sigma_i u_i \quad (i = 1, \dots, r), \quad \text{and } A v_i = 0 \quad (i > r)$$

This reveals the geometric action of any linear map $A: \mathbb{R}^n \to \mathbb{R}^m$:
1. **$V^T$ (Rotation / Reflection in Domain):** Projects an input vector $x \in \mathbb{R}^n$ onto the orthonormal basis $\{v_i\}$, giving coordinates $v_i^T x$.
2. **$\Sigma$ (Axis-Aligned Scaling):** Stretches each coordinate $i$ by the factor $\sigma_i$.
3. **$U$ (Rotation / Reflection in Codomain):** Recombines the stretched coordinates along the output orthonormal basis vectors $\{u_i\}$ in $\mathbb{R}^m$.

Under $A$, the unit sphere $\mathcal{S} = \{x \in \mathbb{R}^n : \|x\|_2 = 1\}$ is mapped onto a solid **hyperellipsoid** in $\mathbb{R}^m$:
- The semi-axes of the ellipsoid align with the directions of the left singular vectors $u_i$.
- The half-lengths of the semi-axes are given by the singular values $\sigma_i$.
- The pre-images of these semi-axes on the domain unit sphere are the right singular vectors $v_i$.

```
Domain Space ℝⁿ                              Codomain Space ℝᵐ
  ┌─────────────┐                              ┌─────────────┐
  │      ↑ v₂   │                              │   ↑ σ₂ u₂   │
  │   ───┼───→  │    ─── Linear Map A ───►     │ ──────┼──────→│
  │      │   v₁ │                              │       │   σ₁ u₁
  │ (Unit Sphere)│                             │(Hyperellipsoid)│
  └─────────────┘                              └─────────────┘
```

---

#### 4. Dyadic (Outer Product) Representation

The SVD can be written as a sum of rank-1 outer products (dyads):
$$A = \sum_{i=1}^r \sigma_i \, u_i v_i^T$$
- Each term $u_i v_i^T \in \mathbb{R}^{m \times n}$ is an elementary matrix of rank 1 with unit Frobenius norm: $\|u_i v_i^T\|_F = \|u_i\|_2 \|v_i\|_2 = 1$.
- These dyads are mutually orthogonal under the Frobenius inner product: $\langle u_i v_i^T, u_j v_j^T \rangle_F = (u_i^T u_j)(v_i^T v_j) = \delta_{ij}$.
- $u_i$ dictates the **column profile** (vertical variation), $v_i^T$ dictates the **row profile** (horizontal variation), and $\sigma_i$ is the **energy / amplitude weight**.

---

### Question 5 — Why are singular values in descending order? Why do small values correspond to less important directions? Why are exact zeros rare?

**1. Why singular values appear in descending order ($\sigma_1 \geq \sigma_2 \geq \dots \geq 0$):**
- **Variational / Energy Optimization:** Singular values are mathematically defined via the Rayleigh quotient as successive maxima of the operator norm:
  $$\sigma_1 = \max_{\|x\|_2=1} \|A x\|_2 = \|A v_1\|_2$$
  $$\sigma_k = \max_{\substack{\|x\|_2=1 \\ x \perp \{v_1, \dots, v_{k-1}\}}} \|A x\|_2 = \|A v_k\|_2$$
  By definition, $\sigma_1$ captures the direction of maximum amplification (variance / stretch), $\sigma_2$ captures the next largest orthogonal stretch, and so forth.
- **Algorithmic convention:** Standard SVD algorithms (e.g., Golub-Kahan bidiagonalization with QR/divide-and-conquer) naturally sort the computed singular values in non-increasing order.

**2. Why small singular values correspond to "less important" directions:**
- **Frobenius Energy:** The total Frobenius energy of the matrix satisfies:
  $$\|A\|_F^2 = \operatorname{Tr}(A^T A) = \sum_{i=1}^r \sigma_i^2$$
  The relative energy carried by the $i$-th component is $\sigma_i^2 / \sum_j \sigma_j^2$. A small $\sigma_i$ contributes negligibly to the overall norm of $A$.
- **Spectral Error:** Truncating directions with singular values $\sigma_{k+1}, \dots, \sigma_r$ introduces a spectral (worst-case operator) error of only $\|A - A_k\|_2 = \sigma_{k+1}$.
- **Noise vs Signal:** In empirical datasets, large singular values capture dominant structural trends, whereas small singular values represent isotropic measurement noise, numerical artifacts, or minor perturbations.

**3. Why floating-point arithmetic makes exact zeros rare:**
- **Full Rank of Continuous Distributions:** For a generic random matrix (entries drawn from standard normal $\mathcal{N}(0,1)$), the columns are linearly independent with probability 1, so all $\min(m,n)$ singular values are strictly positive.
- **Machine Precision Perturbations:** In IEEE 754 double-precision arithmetic, roundoff errors introduce perturbations $\Delta A$ with $\|\Delta A\| \sim \epsilon_{\text{mach}} \|A\| \approx 10^{-16} \|A\|$. By **Weyl's perturbation theorem for singular values**:
  $$|\tilde{\sigma}_i - \sigma_i| \leq \|\Delta A\|_2 \sim O(10^{-16})$$
  Even if a matrix is analytically rank-deficient with theoretical zeros ($\sigma_i = 0$), finite precision perturbs these zeros to small positive quantities $\sim 10^{-15} - 10^{-16}$.
- **Numerical Rank:** Because exact zeros almost never occur in floating-point computations, rank is defined numerically using a tolerance threshold:
  $$\text{rank}_{\text{num}}(A) = \#\{\sigma_i : \sigma_i > \text{tol}\}, \qquad \text{tol} = \max(m,n) \cdot \epsilon_{\text{mach}} \cdot \sigma_1$$

---

## Exercise 2: Best Rank-$k$ Approximation

**Experimental results (same $A \in \mathbb{R}^{10 \times 6}$):**

| $k$ | Measured Error $\|A - A_k\|_F$ | Eckart–Young Theoretical Bound $\sqrt{\sum_{i=k+1}^6 \sigma_i^2}$ | Spectral Error $\|A - A_k\|_2 = \sigma_{k+1}$ |
|:---:|:---:|:---:|:---:|
| **1** | **5.4396** | **5.4396** | 3.7391 |
| **2** | **3.9507** | **3.9507** | 2.7217 |
| **3** | **2.8636** | **2.8636** | 1.8981 |
| **4** | **2.1442** | **2.1442** | 1.5907 |
| **5** | **1.4378** | **1.4378** | 1.4378 |
| **6** | **0.0000** | **0.0000** | 0.0000 |

*The measured numerical error matches the theoretical Eckart–Young bound to 15 decimal places at every $k$.*

---

### Question 4 — Why does SVD give the optimal rank-$k$ approximation?

This optimality is established by the **Eckart–Young–Mirsky Theorem** (1936 / 1960).

#### Theorem Statement:
Let $A \in \mathbb{R}^{m \times n}$ have singular value decomposition $A = \sum_{i=1}^r \sigma_i u_i v_i^T$. For any $k < r$, the truncated SVD matrix:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$
is the global minimizer of the distance $\|A - M\|$ over the manifold of all matrices $M \in \mathbb{R}^{m \times n}$ of rank at most $k$, for both the **Spectral norm** and the **Frobenius norm** (and indeed for any unitarily invariant norm):
$$\min_{\operatorname{rank}(M) \leq k} \|A - M\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$
$$\min_{\operatorname{rank}(M) \leq k} \|A - M\|_2 = \|A - A_k\|_2 = \sigma_{k+1}$$

#### Mathematical Proof / Intuition:

1. **Frobenius Norm Optimality:**
   Because $\{u_i v_j^T\}_{i,j}$ forms an orthonormal basis for $\mathbb{R}^{m \times n}$ equipped with the Frobenius inner product $\langle X, Y \rangle_F = \operatorname{Tr}(X^T Y)$, the squared Frobenius norm decomposes independently:
   $$\|A - A_k\|_F^2 = \left\| \sum_{i=k+1}^r \sigma_i u_i v_i^T \right\|_F^2 = \sum_{i=k+1}^r \sigma_i^2 \|u_i v_i^T\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$
   Since $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r$, keeping the $k$ largest terms $\sigma_1, \dots, \sigma_k$ greedily and globally minimizes the remaining tail sum $\sum_{i=k+1}^r \sigma_i^2$.

2. **Spectral Norm Optimality (Dimension Counting Argument):**
   Let $M$ be any matrix with $\operatorname{rank}(M) \leq k$. By the rank-nullity theorem, the nullspace $\ker(M) \subseteq \mathbb{R}^n$ has dimension $\dim(\ker(M)) = n - \operatorname{rank}(M) \geq n - k$.
   Consider the subspace $\mathcal{V}_{k+1} = \operatorname{span}(v_1, v_2, \dots, v_{k+1}) \subseteq \mathbb{R}^n$, which has dimension $k+1$.
   By the dimension theorem for intersecting subspaces:
   $$\dim(\ker(M) \cap \mathcal{V}_{k+1}) \geq (n - k) + (k + 1) - n = 1$$
   Therefore, there exists a unit vector $z \in \ker(M) \cap \mathcal{V}_{k+1}$ ($\|z\|_2 = 1$). For this vector:
   $$M z = 0 \implies \|(A - M)z\|_2 = \|Az\|_2$$
   Since $z = \sum_{i=1}^{k+1} c_i v_i$ with $\sum c_i^2 = 1$, we have:
   $$\|Az\|_2^2 = \left\| \sum_{i=1}^{k+1} c_i \sigma_i u_i \right\|_2^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \geq \sigma_{k+1}^2 \sum_{i=1}^{k+1} c_i^2 = \sigma_{k+1}^2$$
   Thus, $\|A - M\|_2 \geq \|(A - M)z\|_2 \geq \sigma_{k+1} = \|A - A_k\|_2$. Hence $A_k$ is the global minimizer.

**Practical Implication:** SVD provides the theoretical lower bound on reconstruction error for any dimensionality reduction, matrix compression, or latent factor modeling scheme.

---

## Exercise 3: Image Compression with SVD

**Image:** `cameraman`, grayscale $X \in \mathbb{R}^{512 \times 512}$, full rank 512, uncompressed size = $512 \times 512 = 262{,}144$ scalars.

| $k$ | Energy Captured $\frac{\|X_k\|_F^2}{\|X\|_F^2}$ | Storage $k(2n+1)$ | Compression Factor $c_k$ | Frobenius Error $\|X - X_k\|_F$ | Visual Appearance |
|:---:|:---:|:---:|:---:|:---:|:---|
| **2** | 92.03% | 2,050 | 99.22% | 21,474.7 | Coarse illumination gradient only; no edges or recognizable shapes. |
| **20** | 98.98% | 20,500 | 92.18% | 7,699.9 | Recognizable silhouette and coarse structure; textures blurred. |
| **50** | 99.60% | 51,250 | 80.45% | 4,836.1 | Sharp edges, clear silhouette, tripod, coat details visible. |
| **100** | 99.85% | 102,500 | 60.90% | 2,992.1 | Nearly indistinguishable from original; only subtle background texture lost. |
| **200** | 99.97% | 205,000 | 21.80% | 1,342.4 | Visually identical to original. |

---

### Deep Dive: What do $U$ and $V$ Represent in Image Compression?

An image $X \in \mathbb{R}^{m \times n}$ is a 2D spatial grid where rows represent horizontal scan lines and columns represent vertical pixel columns.

```
                      X ≈ σ₁ (u₁ ⊗ v₁ᵀ) + σ₂ (u₂ ⊗ v₂ᵀ) + ⋯ + σₖ (uₖ ⊗ vₖᵀ)
                         ▲              ▲                     ▲
                         │              │                     │
                    Base background    Coarse edges      Fine textures & noise
```

1. **$U = [u_1, u_2, \dots, u_m] \in \mathbb{R}^{m \times m}$ (Vertical Basis Patterns):**
   - Each column $u_i \in \mathbb{R}^m$ is a 1D vertical profile (basis function for the image columns).
   - $u_1$ represents the average vertical intensity distribution (smooth vertical gradient).
   - Higher $u_i$ represent increasingly high-frequency vertical oscillations (vertical edges, textures, ripples).

2. **$V = [v_1, v_2, \dots, v_n] \in \mathbb{R}^{n \times n}$ (Horizontal Basis Patterns):**
   - Each column $v_i \in \mathbb{R}^n$ (or row $v_i^T$) is a 1D horizontal profile (basis function for the image rows).
   - $v_1$ represents the average horizontal intensity distribution.
   - Higher $v_i$ represent high-frequency horizontal oscillations (horizontal edges, textures).

3. **The Dyad $u_i v_i^T \in \mathbb{R}^{m \times n}$ (Rank-1 "Eigen-Image"):**
   - The outer product $u_i v_i^T$ creates a 2D separable grid pattern.
   - $\sigma_1 u_1 v_1^T$: Captures the global illumination and background contrast (92.03% of total energy).
   - $\sigma_i u_i v_i^T$ ($i=2,\dots,50$): Add structured geometric lines, silhouette boundaries, and intermediate textures.
   - $\sigma_i u_i v_i^T$ ($i > 100$): Capture fine pixel-level variations and high-frequency noise.

---

### Question 4 — Visual Quality, Energy Concentration, Compression Trade-off, and Eckart–Young Connection

1. **Visual Quality Progression:**
   Visual quality improves non-linearly. The perceptual gain per added component is huge from $k=2$ to $k=50$ (where human-recognizable shapes, edges, and silhouettes emerge) and shows diminishing marginal returns beyond $k=100$.

2. **Why Most Energy is Concentrated in the First Few Singular Values:**
   Natural images possess strong **spatial auto-correlation**: adjacent pixels have similar luminance values. This smoothness causes the spatial covariance to have a small number of dominant eigenvalues. Consequently, the singular value spectrum $\sigma_i$ decays exponentially fast, concentrating $>98.98\%$ of the Frobenius energy in the top 20 components.

3. **Trade-off Between Compression and Fidelity:**
   Storing $X_k$ requires storing $k$ left vectors ($k \times m$), $k$ singular values ($k$), and $k$ right vectors ($k \times n$), yielding a total of $k(m+n+1)$ numbers. The compression factor:
   $$c_k = 1 - \frac{k(m+n+1)}{mn} = 1 - \frac{k(1025)}{262144}$$
   represents the fraction of storage saved. At $k=50$, we achieve $80.45\%$ memory savings with $99.60\%$ energy retention, representing the optimal operational knee of the rate-distortion curve.

4. **Connection to Optimal Low-Rank Approximation:**
   The Eckart–Young theorem guarantees that no other matrix factorization of rank $k$ (e.g., Fourier, Wavelet, or random projections) can achieve lower Frobenius error for the same rank $k$.

---

## Exercise 4/5: PCA + Clustering on MNIST

**Dataset:** MNIST ($N = 42{,}000$ images, $d = 784$ features from $28 \times 28$ pixels).

**PCA ($k=2$) Experimental Results on Digit Pairs:**

| Digit Pair | $N_{\text{train}}$ | $N_{\text{test}}$ | Variance Explained ($k=2$) | Centroid Distance $\|\mu_a - \mu_b\|_2$ | Cluster Overlap in 2D |
|:---:|:---:|:---:|:---:|:---:|:---|
| **1 vs 7** | 7,268 | 1,817 | **36.61%** | **1,358.94** | Minimal overlap; two distinct, well-separated clusters. |
| **3 vs 4** | 6,738 | 1,685 | **24.19%** | **1,330.64** | Clean separation with a clear decision corridor. |
| **2 vs 8** | 6,592 | 1,648 | **18.82%** | **978.26** | Significant overlap; dense shared region in center. |
| **3 vs 4 ($k=3$)** | 6,738 | 1,685 | **29.88%** | — | 3D projection resolves depth ambiguity and separates borderline samples. |

---

### Deep Dive: What do $U$, $\Sigma$, and $V$ Represent in PCA?

Let $X_{\text{train}} \in \mathbb{R}^{N \times d}$ be the training data matrix ($N$ samples, $d = 784$ pixels).
1. **Centering:** Compute mean row vector $c(X_{\text{train}}) = \mu = \frac{1}{N} \mathbf{1}^T X_{\text{train}} \in \mathbb{R}^{1 \times d}$.
   The centered matrix is $X_c = X_{\text{train}} - \mathbf{1}\mu \in \mathbb{R}^{N \times d}$.
2. **SVD of Centered Data:**
   $$X_c = U \Sigma V^T$$
   where $U \in \mathbb{R}^{N \times d}$ (thin SVD), $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_d) \in \mathbb{R}^{d \times d}$, and $V \in \mathbb{R}^{d \times d}$.

```
Data Space (Pixels) ℝ⁷⁸⁴                        Sample Space ℝᴺ
      ┌─────────────┐                               ┌─────────────┐
      │  V columns  │                               │  U columns  │
      │  v₁, v₂, ⋯  │                               │  u₁, u₂, ⋯  │
      │(Eigen-Digits)                               │(Sample Coords)
      └─────────────┘                               └─────────────┘
             ▲                                             ▲
             │                                             │
   Feature space basis                           Normalized scores of
  (pixel loading patterns)                       N images along PC axes
```

---

#### 1. What does $V$ represent in PCA? (Principal Directions / Loadings)

- **Eigenvectors of the Sample Covariance Matrix:**
  The sample covariance matrix $C \in \mathbb{R}^{d \times d}$ is:
  $$C = \frac{1}{N-1} X_c^T X_c = \frac{1}{N-1} (V \Sigma^T U^T)(U \Sigma V^T) = V \left( \frac{\Sigma^2}{N-1} \right) V^T$$
  The columns of $V = [v_1, v_2, \dots, v_d]$ are the **orthonormal eigenvectors of the covariance matrix $C$**.
- **Physical Meaning in Pixel Space ($\mathbb{R}^{784}$):**
  - Each vector $v_i \in \mathbb{R}^{784}$ is a **principal axis / loading vector** (often visualized as an "eigen-digit" when reshaped to $28 \times 28$).
  - $v_1$: Direction in 784D space along which the dataset has maximum sample variance.
  - $v_2$: Direction orthogonal to $v_1$ that captures the second largest variance.
- **Loading Weights:** The $j$-th element of $v_i$ indicates how much pixel $j$ contributes to the $i$-th principal component.

---

#### 2. What does $U$ represent in PCA? (Normalized Sample Coordinates / Scores)

- **Eigenvectors of the Sample Gram Matrix:**
  The Gram matrix $G = X_c X_c^T \in \mathbb{R}^{N \times N}$ represents sample-to-sample dot products:
  $$G = X_c X_c^T = (U \Sigma V^T)(V \Sigma U^T) = U \Sigma^2 U^T$$
  The columns of $U = [u_1, u_2, \dots, u_d]$ are the **orthonormal eigenvectors of the Gram matrix $G$**.
- **Physical Meaning in Sample Space ($\mathbb{R}^N$):**
  - Each column $u_i \in \mathbb{R}^N$ contains the normalized coordinates of all $N$ data samples along the $i$-th principal axis.
  - The $j$-th entry $u_{j,i}$ is the coordinate of image $j$ along PC $i$, normalized so that $\|u_i\|_2 = 1$.

---

#### 3. What does $\Sigma$ represent in PCA? (Singular Values and Variance)

- The eigenvalues of the covariance matrix $C$ are $\lambda_i = \frac{\sigma_i^2}{N-1} = \operatorname{Var}(z_i)$.
- The singular value $\sigma_i$ is proportional to the **standard deviation** of the dataset projected along the $i$-th principal axis:
  $$\sigma_i = \sqrt{N-1} \cdot \operatorname{std}(z_i)$$
- **Explained Variance Ratio:** The fraction of total variance captured by the top $k$ principal components is:
  $$\text{EVR}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}$$

---

#### 4. The Projected Data Matrix $Z$: The Bridge $X_c V = U \Sigma$

Multiplying $X_c = U \Sigma V^T$ on the right by $V_k \in \mathbb{R}^{d \times k}$ (the first $k$ columns of $V$):
$$Z_{\text{train}} = X_c V_k = U_k \Sigma_k \in \mathbb{R}^{N \times k}$$

This reveals a profound dual identity:
- **Projection view ($X_c V_k$):** We project the 784D centered data points onto the principal component axes $V_k$.
- **Spectral view ($U_k \Sigma_k$):** The projected coordinates are directly given by the left singular vectors $U_k$ scaled by the singular values $\Sigma_k$.

---

### Why Projecting the Test Set Requires $c(X_{\text{train}})$ (Data Leakage & Geometric Consistency)

For test samples $X_{\text{test}} \in \mathbb{R}^{N_{\text{test}} \times d}$, projection is defined as:
$$Z_{\text{test}} = (X_{\text{test}} - \mathbf{1} c(X_{\text{train}})) V_k$$

**Why we MUST center using the training mean $c(X_{\text{train}})$ and NOT the test mean:**
1. **Affine Coordinate System:** The principal axes $V_k$ were computed relative to the training origin $c(X_{\text{train}})$. Subtracting $c(X_{\text{train}})$ shifts test points into this exact coordinate frame. Subtracting $c(X_{\text{test}})$ would translate the test points relative to an arbitrary offset, displacing the test cluster away from its corresponding training cluster.
2. **Preventing Data Leakage:** In real-world machine learning, the test set represents unseen data that may arrive as a single sample ($N_{\text{test}} = 1$), for which a sample mean cannot be computed. All preprocessing parameters (mean, scaling, projection matrices) must be learned strictly on the training partition.

---

### Questions on Cluster Separation, Digit Similarity, and Effect of $k=3$

1. **What Determines Cluster Separation in PCA Space:**
   PCA is an **unsupervised** technique: it finds directions that maximize *total variance* without using class labels. Separation occurs when the *inter-class variance* (difference between digit mean shapes) aligns with the directions of maximum total variance.
   - **1 vs 7:** Very high explained variance (36.61%) and largest centroid separation (1,358.94). Digit 1 is a single vertical stroke, while 7 has a prominent horizontal bar and diagonal slant. The pixel covariance between these shapes is orthogonal, allowing the top 2 PCs to isolate the classes cleanly.
   - **3 vs 4:** Clear separation (variance 24.19%, distance 1,330.64). The curved loops of 3 differ fundamentally from the straight intersections of 4.
   - **2 vs 8:** Low explained variance (18.82%) and smallest centroid separation (978.26). Digits 2 and 8 share similar curved top loops and bottom horizontal strokes. The dominant variance modes reflect shared handwriting style (thickness, slant) rather than class differences, resulting in overlapping clusters in 2D.

2. **Effect of Increasing $k$ to 3:**
   For digits 3 vs 4, explained variance increases from $24.19\%$ ($k=2$) to $29.88\%$ ($k=3$). In the 3D scatterplot, the third principal component $z_3$ resolves depth ambiguities: points that overlapped in the $(z_1, z_2)$ plane are separated along $z_3$.

---

## Exercise 6: Classification After PCA

**Experimental Results ($k=2$ PCA Features, 80/20 Train/Test Split):**

| Digit Pair | Classifier | Test Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TP, FP, FN, TN) | Centroid Distance in $\mathbb{R}^2$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **3 vs 4** | **Logistic Regression (PCA)** | **0.9757** | 0.9592 | 0.9928 | **0.9757** | $\text{TP}=822, \text{FP}=35, \text{FN}=6, \text{TN}=822$ | **1,330.64** |
| | **Nearest-Centroid (PCA)** | **0.9757** | 0.9613 | 0.9903 | **0.9756** | $\text{TP}=820, \text{FP}=33, \text{FN}=8, \text{TN}=824$ | |
| **1 vs 7** | **Logistic Regression (PCA)** | **0.9697** | 0.9963 | 0.9400 | **0.9674** | $\text{TP}=815, \text{FP}=3, \text{FN}=52, \text{TN}=947$ | **1,364.62** |
| | **Nearest-Centroid (PCA)** | **0.9664** | 0.9963 | 0.9331 | **0.9637** | $\text{TP}=809, \text{FP}=3, \text{FN}=58, \text{TN}=947$ | |
| **2 vs 3** | **Logistic Regression (PCA)** | **0.9045** | 0.8956 | 0.9136 | **0.9045** | $\text{TP}=772, \text{FP}=90, \text{FN}=73, \text{TN}=771$ | **1,090.69** |
| | **Nearest-Centroid (PCA)** | **0.9009** | 0.8885 | 0.9148 | **0.9015** | $\text{TP}=773, \text{FP}=97, \text{FN}=72, \text{TN}=764$ | |

---

### Step 4 — Linear Classifier vs Nearest-Centroid Classifier

#### 1. Mathematical Derivation of Decision Boundaries

- **Logistic Regression Classifier:**
  Given projected features $z = (z_1, z_2) \in \mathbb{R}^2$ and augmented vector $\tilde{z} = [1, z_1, z_2]^T$, the model predicts $P(y=1|z) = \sigma(\Theta^T \tilde{z})$. The decision threshold $P \geq 0.5$ yields the linear boundary:
  $$\Theta^T \tilde{z} = \theta_0 + \theta_1 z_1 + \theta_2 z_2 = 0 \quad \implies \quad z_2 = -\frac{\theta_1}{\theta_2} z_1 - \frac{\theta_0}{\theta_2}$$
  This is a straight line in $\mathbb{R}^2$. The parameters $\Theta$ are optimized via SGD to minimize the empirical Binary Cross-Entropy (BCE) loss.

- **Nearest-Centroid Classifier:**
  Let $\mu_0, \mu_1 \in \mathbb{R}^2$ be the empirical class centroids in PCA space. A test point $z$ is assigned to class 1 if $\|z - \mu_1\|_2^2 < \|z - \mu_0\|_2^2$:
  $$\|z - \mu_1\|_2^2 - \|z - \mu_0\|_2^2 < 0$$
  $$(z^T z - 2 \mu_1^T z + \|\mu_1\|_2^2) - (z^T z - 2 \mu_0^T z + \|\mu_0\|_2^2) < 0$$
  $$2 (\mu_0 - \mu_1)^T z + (\|\mu_1\|_2^2 - \|\mu_0\|_2^2) < 0$$
  The decision boundary is the equality condition:
  $$(\mu_1 - \mu_0)^T \left( z - \frac{\mu_0 + \mu_1}{2} \right) = 0$$
  **Geometric Proof:** This is explicitly the equation of the **perpendicular bisector** (orthogonal hyperplane) of the line segment joining the two centroids $\mu_0$ and $\mu_1$, passing through their midpoint $\frac{\mu_0 + \mu_1}{2}$.

```
PCA Plane (z₁, z₂)
           Digit A (•)                 Digit B (•)
             •  •  •                     •  •  •
            •  μ_A  •       |           •  μ_B  •
             •  •  •        |            •  •  •
                            |  Perpendicular
                            |  Bisector Line
             ───────────────┼───────────────►
                         Midpoint
```

#### 2. Theoretical Equivalence and Differences

- **When they are identical (Isotropic Gaussian Clusters):**
  If the two digit classes in PCA space follow spherical Gaussian distributions with equal variance $\Sigma_0 = \Sigma_1 = \sigma^2 I_2$ and equal class priors $P(y=0) = P(y=1) = 0.5$, then the **Bayes optimal classifier** is analytically identical to the nearest-centroid perpendicular bisector.
- **When they diverge (Heteroscedastic / Asymmetric Clusters):**
  - **1 vs 7 (Precision 0.9963 vs Recall 0.9400):** Digit 1 is geometrically rigid, forming a dense, compact cluster with small variance. Digit 7 exhibits high intra-class variability (slants, serifs, cross-bars), forming an elongated, diffuse cluster. Logistic regression places the decision boundary closer to the dense cluster (1) to eliminate false alarms for 1 (yielding $\text{FP}=3$), which causes some diffuse 7s to fall across the boundary ($\text{FN}=52$).
  - **2 vs 3 (Accuracy ~90.4%):** Visual similarity results in heavy cluster overlap along both $z_1$ and $z_2$. Because the overlap region is non-separable linearly, both classifiers achieve ~90% accuracy, reflecting the Bayes error rate of linear separation in 2D.

---

### Back-Projection: Decision Boundaries in Original 784D Pixel Space

In 2D PCA space, the linear boundary is:
$$\theta_0 + \theta_1 z_1 + \theta_2 z_2 = \theta_0 + \theta_{1:2}^T z = 0$$
Substituting $z = (x - \mu) V_2$ (where $V_2 = [v_1, v_2] \in \mathbb{R}^{d \times 2}$):
$$\theta_0 + \theta_{1:2}^T V_2^T (x - \mu)^T = 0 \quad \iff \quad \underbrace{(V_2 \theta_{1:2})^T}_{w^T \in \mathbb{R}^{1 \times 784}} x^T + \underbrace{(\theta_0 - \theta_{1:2}^T V_2^T \mu^T)}_{b \in \mathbb{R}} = 0$$

$$w^T x + b = 0, \qquad \text{where } w = \theta_1 v_1 + \theta_2 v_2 \in \mathbb{R}^{784}$$

**Fundamental Insight (Subspace Regularization):**
A linear classifier trained on 2D PCA features corresponds to an exact **hyperplane classifier in the original 784-dimensional pixel space**. However, the normal vector $w$ is strictly constrained to lie in the 2D subspace spanned by $\{v_1, v_2\}$. PCA dimensionality reduction acts as a **structural linear regularizer**, preventing overfitting in 784 dimensions by eliminating noisy, orthogonal feature directions.

---

## Summary of Key Theoretical Takeaways

| Mathematical Concept | Core Equation / Definition | Role of $U$ | Role of $V$ | Key Theoretical Takeaway |
|---|---|---|---|---|
| **SVD Factorization** | $A = U \Sigma V^T = \sum_{i=1}^r \sigma_i u_i v_i^T$ | Columns $u_i$ are orthonormal basis for $\operatorname{Col}(A)$ (eigenvectors of $A A^T$) | Columns $v_i$ are orthonormal basis for $\operatorname{Row}(A)$ (eigenvectors of $A^T A$) | Maps domain unit sphere to codomain hyperellipsoid without cross-talk |
| **Eckart–Young Theorem** | $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ | Governs optimal codomain rank-$k$ subspace | Governs optimal domain rank-$k$ subspace | Globally optimal low-rank matrix approximation in Frobenius and 2-norm |
| **Numerical Rank** | $\text{rank}_{\text{num}} = \#\{\sigma_i > \text{tol}\}$ | Left singular vectors corresponding to signal | Right singular vectors corresponding to signal | Roundoff errors $\sim 10^{-16}$ prevent exact zeros; requires threshold $\text{tol}$ |
| **Image Compression** | $X_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ | Vertical basis patterns (column profiles) | Horizontal basis patterns (row profiles) | Spatial correlation causes rapid spectral decay; $k=50$ captures $>99.6\%$ energy |
| **PCA Dimensionality Reduction** | $X_c = U \Sigma V^T \implies Z = X_c V_k = U_k \Sigma_k$ | Normalized sample scores $u_i \in \mathbb{R}^N$ (eigenvectors of $X_c X_c^T$) | Principal component loading axes $v_i \in \mathbb{R}^d$ (eigenvectors of covariance $C$) | Unsupervised variance maximization; projects $d=784$ features onto top $k$ principal axes |
| **Test Set Projection** | $Z_{\text{test}} = (X_{\text{test}} - \mathbf{1} c(X_{\text{train}})) V_k$ | — | Learned projection matrix $V_k$ applied to test data | Must use training centroid $c(X_{\text{train}})$ to prevent data leakage and maintain origin |
| **Nearest-Centroid Classifier** | $\arg\min_c \|z - \mu_c\|_2^2$ | — | — | Decision boundary is the perpendicular bisector line between class centroids |
| **Logistic vs Centroid** | $\sigma(\Theta^T \tilde{z}) \geq 0.5 \iff \Theta^T \tilde{z} \geq 0$ | — | — | Equivalent for spherical equal-covariance Gaussians; logistic adapts to asymmetric shapes |
| **Subspace Regularization** | $w = V_k \theta_{1:k} \in \mathbb{R}^{784}$ | — | Constrains high-dimensional weight vector $w$ to $\operatorname{span}(V_k)$ | Prevents high-dimensional overfitting by restricting boundary to principal subspace |
