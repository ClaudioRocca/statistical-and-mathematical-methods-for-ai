# Homework 2 – Stochastic Gradient Descent: Theoretical Answers
**Author:** Claudio Rocca

---

## Exercise 1: SGD vs GD on a Simple 1D Regression Problem

**Experimental results (seed 42, 30 epochs, lr=0.1):**

| Method | Final Loss | Θ₀ | Θ₁ |
|---|---|---|---|
| GD | 0.0496 | 1.4095 | 1.2374 |
| SGD batch=1 | 0.0001 | 0.9981 | 1.9991 |
| SGD batch=10 | 0.0001 | 0.9995 | 2.0024 |
| SGD batch=50 | 0.0046 | 1.1242 | 1.7706 |
| SGD batch=200 (≡ GD) | 0.0496 | 1.4095 | 1.2374 |

True parameters: Θ₀=1, Θ₁=2.

---

### Question 4 — Discuss: GD smoothness vs SGD noise, and the effect of batch size.

**Why GD is smooth but slow for large N:**

Full GD computes the exact gradient over all N=200 samples at every epoch:
$$\nabla\mathcal{L}(\Theta) = \frac{2}{N}\sum_{i=1}^N (f_\Theta(x^{(i)}) - y^{(i)})\,\tilde{x}^{(i)}$$

This is a deterministic quantity — the same gradient is computed at every step given the same $\Theta$. As a result, the loss curve decreases monotonically and smoothly, with no iteration-to-iteration fluctuations. The trajectory in $(\Theta_0, \Theta_1)$ space follows a clean descent path.

The cost is that each epoch requires scanning all N samples. More critically, with lr=0.1 and only 30 epochs, GD has not converged: the results show a final loss of 0.0496 and parameters (1.41, 1.24) still far from (1.0, 2.0). This is because 30 full-batch steps at this learning rate are not enough updates to follow the loss surface to its minimum — especially relevant when N is large and each gradient evaluation is expensive.

**Why SGD is noisy but progresses faster:**

SGD replaces the exact gradient with a stochastic estimate computed on a mini-batch $\mathcal{M}_k$:
$$g_k = \frac{1}{N_\text{batch}}\sum_{i \in \mathcal{M}_k} \nabla\ell_i(\Theta)$$

This estimate is **unbiased** ($\mathbb{E}[g_k] = \nabla\mathcal{L}(\Theta)$) but has nonzero variance. Within a single epoch, SGD performs $N/N_\text{batch}$ parameter updates instead of one — effectively doing more gradient steps per epoch. With batch=1, that is 200 updates per epoch vs 1 for GD. In 30 epochs, SGD batch=1 performs 6000 updates, driving the loss to ~0.0001 and recovering parameters extremely close to the truth (0.998, 1.999).

The noise in the gradient estimate causes the loss curve to oscillate rather than monotonically decrease, and the parameter trajectory is jagged rather than smooth. However, this noise can be beneficial: it prevents the optimizer from getting stuck in flat regions and helps explore the parameter space more efficiently.

**How batch size affects noise and convergence stability:**

- **batch=1:** maximum noise, fastest effective progress per epoch (200 updates), loss curve oscillates strongly, but converges closest to the true parameters in 30 epochs.
- **batch=10:** reduced noise (variance ∝ 1/batch_size, as confirmed in Exercise 2), still 20 updates per epoch, converges to comparable quality (loss 0.0001) with a smoother trajectory.
- **batch=50:** only 4 updates per epoch, significantly less progress. Final loss 0.0046 and parameters (1.12, 1.77) — still far from the truth. The reduced noise comes at the cost of fewer updates per epoch.
- **batch=200 (≡ GD):** identical to full GD — deterministic, smooth, but only 1 update per epoch and the same slow convergence.

The key insight is that there is a **variance–computation tradeoff**: small batches → high variance but many cheap updates per epoch; large batches → low variance but fewer expensive updates per epoch. For a fixed epoch budget, small batches often win on simple problems. For very large N, SGD's per-iteration cost advantage becomes decisive.

---

## Exercise 2: Variance of the Stochastic Gradient

**Experimental results (K=100 batches at Θ=(1,2)):**

| Batch size | Var(g) |
|---|---|
| 1 | 0.037538 |
| 5 | 0.007559 |
| 20 | 0.002120 |
| 200 (full) | 0.000000 |

---

### Question 5 — Discuss: variance decrease, stability, and the stability–cost tradeoff.

**Why variance decreases with larger batches:**

The stochastic gradient is an average of $N_\text{batch}$ i.i.d. sample gradients:
$$g_k = \frac{1}{N_\text{batch}}\sum_{i \in \mathcal{M}_k} \nabla\ell_i(\Theta)$$

By the properties of variance for independent random variables:
$$\text{Var}(g_k) = \frac{1}{N_\text{batch}}\,\text{Var}(\nabla\ell_i(\Theta))$$

The variance scales as $1/N_\text{batch}$. The results confirm this: going from batch=1 to batch=5 reduces variance by approximately 5× (0.0375 → 0.0076), and from 1 to 20 by approximately 18× (0.0375 → 0.0021), consistent with the theoretical $1/N_\text{batch}$ scaling. At batch=N=200, the gradient is exact and variance is zero (up to floating point).

**Why SGD becomes more stable as $N_\text{batch}$ increases:**

With high variance, successive gradient estimates point in very different directions, causing the parameter updates to oscillate. This makes the loss curve noisy and can cause the iterate to overshoot or wander. As batch size grows, the gradient estimate more reliably points in the direction of steepest descent of the true loss, producing smoother and more predictable updates. This directly translates to the smoother loss curves and parameter trajectories observed in Exercise 1 for larger batches.

**The stability–computational cost tradeoff:**

A larger batch requires evaluating more sample gradients per step, costing $O(N_\text{batch})$ floating-point operations. Halving the variance requires doubling the batch size — a linear cost for a square-root benefit in noise standard deviation. In practice this means: past a certain batch size, the marginal improvement in gradient quality does not justify the additional compute. Modern deep learning practice uses moderate batch sizes (32–256) that balance noise (sufficient to escape shallow local minima and generalise better) against per-step cost.

---

## Exercise 3: SGD in 2D on a Non-Convex Function

$$\mathcal{L}(\Theta_1, \Theta_2) = (\Theta_1^2 - 1)^2 + 10(\Theta_2 - \Theta_1^2)^2$$

**Stationary points:** The function has two global minima at $(\pm 1, 1)$ with $\mathcal{L}=0$, a saddle at $(0, 0)$, and two local maxima. Starting from $\Theta_0 = (1, 2)$, all runs converge toward $(+1, 1)$.

**Experimental results (3000 epochs, starting from (1, 2)):**

| σ | lr | Final L | Final Θ |
|---|---|---|---|
| 0.00 | 1e-2 | 0.0000 | (1.000, 1.000) |
| 0.00 | 1e-3 | 0.0001 | (1.004, 1.009) |
| 0.00 | 1e-4 | 0.2592 | (1.221, 1.534) |
| 0.10 | 1e-2 | 0.0001 | (0.996, 0.991) |
| 0.10 | 1e-3 | 0.0001 | (1.004, 1.009) |
| 0.50 | 1e-2 | 0.0033 | (1.024, 1.037) |
| 0.50 | 1e-3 | 0.0004 | (1.010, 1.019) |
| 0.99 | 1e-2 | 0.0011 | (0.987, 0.968) |
| 0.99 | 1e-3 | 0.0001 | (1.001, 1.006) |
| (all σ) | 1e-4 | ~0.26 | (~1.22, ~1.53) |

---

### Question 3 — Discuss: noise helping escape shallow minima, and too much noise preventing convergence.

**How noise helps escape shallow minima or bad regions:**

Pure GD ($\sigma=0$) with lr=1e-2 converges perfectly to the minimum (final L=0.0000) in 3000 iterations. With lr=1e-4, GD stalls at L=0.2592: the step is too small to travel the distance from (1.22, 1.53) to (1, 1) in 3000 steps — a pure step-size problem, not a landscape problem.

The noise term $\varepsilon_k \sim \mathcal{N}(0, \sigma^2 I)$ perturbs the gradient at every step, effectively making the optimizer take random steps in addition to the gradient direction. This can help in two ways:
1. **Escaping saddle points:** near a saddle, the gradient is near zero and GD stalls. Noise injects a random perturbation that can push the iterate off the saddle ridge toward a descent direction.
2. **Crossing shallow barriers:** in non-convex landscapes, noise can allow the iterate to cross low-energy barriers that a pure gradient step would be unable to surmount.

In this experiment, since the starting point (1, 2) is already in the basin of attraction of (1, 1), noise does not meaningfully change which minimum is found — but in a more adversarial initialisation (e.g. near the saddle at (0,0)), noise would be the deciding factor.

**How too much noise prevents convergence:**

With $\sigma=0.5$ and lr=1e-2, the final loss is 0.0033 — worse than $\sigma=0.1$ (0.0001). With $\sigma=0.99$, lr=1e-2 gives 0.0011. Even though these runs converge to the neighbourhood of the minimum, the persistent noise prevents settling precisely at it: once the gradient norm is small (near the minimum), the noise term dominates the update, causing the iterate to wander around the minimum rather than converge to it.

This is the fundamental tension of noisy gradient descent: the noise that helps early in training (exploring, escaping bad regions) becomes harmful late in training (preventing precise convergence). The standard remedy is **learning rate decay** — reducing $\eta_k$ over time so that the effective noise $\eta_k \varepsilon_k$ shrinks to zero, allowing eventual convergence. Note that all runs with lr=1e-4 stall regardless of $\sigma$, confirming that the step size must be large enough to make meaningful progress; noise alone cannot compensate for an inadequate learning rate.

---

## Exercise 4: ML Project with SGD on Insurance Dataset

**Dataset:** 1338 samples, features: age, bmi, children (standardised), target: charges (standardised). Model: linear, lr=1e-2, 1000 epochs.

**Final results:**

| Method | Θ (bias, age, bmi, children) | Final Loss | Final ‖∇L‖ |
|---|---|---|---|
| OLS (analytical) | (0.000, 0.278, 0.167, 0.054) | 0.879902 | — |
| GD | (0.000, 0.278, 0.167, 0.054) | 0.879902 | 0.000000 |
| SGD batch=1 | (0.047, 0.386, 0.167, 0.067) | 0.893984 | 0.2397 |
| SGD batch=10 | (0.003, 0.286, 0.185, 0.042) | 0.880444 | 0.0474 |
| SGD batch=50 | (-0.004, 0.279, 0.165, 0.052) | 0.879926 | 0.0098 |

---

### Question 4 — Discuss: smooth vs noisy curves, batch size effects, convergence region, and SGD scalability.

**Why GD gives a smooth curve and SGD oscillates:**

GD evaluates the exact gradient over all 1338 samples at every epoch, producing a deterministic, monotonically decreasing loss curve. The gradient norm curve likewise decreases smoothly to zero (‖∇L‖ = 0.000000 at convergence).

SGD uses a mini-batch estimate, which is noisy. Even after many epochs, the stochastic gradient evaluated on a random batch may point in a different direction from the true gradient, causing the loss and gradient norm to fluctuate from epoch to epoch. The smaller the batch, the larger the fluctuations: batch=1 shows the most oscillation, batch=50 the least.

**Why larger batches reduce noise but cost more per iteration:**

As shown analytically in Exercise 2, $\text{Var}(g_k) \propto 1/N_\text{batch}$. The final gradient norm confirms this empirically: batch=1 ends at ‖∇L‖=0.2397, batch=10 at 0.0474, batch=50 at 0.0098. Each 5× increase in batch size roughly halves the gradient norm residual. However, each epoch of SGD with batch size $b$ costs $N/b$ gradient evaluations per epoch, each of cost $O(b)$ — so total per-epoch cost is $O(N)$ regardless of batch size. The benefit of small batches is more parameter updates per unit compute, not cheaper epochs.

**Why all methods converge to roughly the same region:**

The MSE loss for a linear model is a strictly convex quadratic in $\Theta$. It has a unique global minimum — the OLS solution $\Theta^* = (X^TX)^{-1}X^TY$. All gradient-based methods, whether deterministic or stochastic, converge to this unique minimiser given sufficient iterations and a suitable learning rate. The OLS solution is (0.000, 0.278, 0.167, 0.054) with loss 0.879902. GD matches it exactly; SGD batch=50 reaches loss 0.879926, nearly identical. SGD batch=1 is furthest (0.893984) due to residual noise preventing precise convergence in 1000 epochs.

The residual loss (~0.88) after convergence represents the irreducible variance in the data that a linear model with only three features cannot explain — principally because `smoker` status (a binary categorical variable excluded here) is the dominant predictor of insurance charges.

**Why SGD is more suitable for large datasets, even when noisy:**

On 1338 samples, GD is not computationally expensive. But consider scaling N to $10^6$ or $10^9$ (typical in modern ML). Each GD epoch requires computing $\nabla\ell_i$ for all N samples before making a single parameter update. SGD with batch=32 makes $N/32 \approx 31,000$ parameter updates per epoch, each using only 32 samples. This means:
1. **Faster initial progress:** useful model performance is reached much earlier in wall-clock time.
2. **Memory efficiency:** only a mini-batch needs to be in memory at once, making it feasible to train on datasets that do not fit in RAM.
3. **Implicit regularisation:** the noise in SGD acts as a form of regularisation, often improving generalisation on unseen data compared to exact GD.
4. **Better escape from bad critical points:** on non-convex losses (neural networks), the stochasticity helps avoid saddle points and sharp minima.

The results confirm that SGD batch=10 and batch=50 reach essentially the same loss as GD (0.880444 vs 0.879902) after 1000 epochs, with the advantage that on larger datasets each epoch would be proportionally cheaper.

---

## Summary of Key Theoretical Takeaways

| Concept | Core Idea |
|---|---|
| **SGD update** | Replace exact gradient with mini-batch estimate: unbiased but noisy |
| **Variance scaling** | $\text{Var}(g_k) \propto 1/N_\text{batch}$: doubling batch halves variance |
| **Noise in non-convex** | Helps escape saddles/shallow minima early; prevents convergence late → use lr decay |
| **Batch size tradeoff** | Small batch: many noisy updates per epoch; large batch: few clean updates |
| **Convex loss** | All methods find the same unique minimum; SGD is noisier but comparably accurate |
| **Scalability** | SGD's key advantage: $O(N_\text{batch})$ cost per update, not $O(N)$ |
