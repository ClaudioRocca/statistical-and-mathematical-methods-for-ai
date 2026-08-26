# Homework 1 – Gradient Descent: Theoretical Answers
**Author:** Claudio Rocca

---

## Exercise 1: GD on a 1D Function

$$\mathcal{L}(\Theta) = (\Theta - 3)^2 + 1$$

---

### Question 1 — Compute the Gradient of $\mathcal{L}(\Theta)$ explicitly.

**Answer:**

Applying the chain rule to the composite function $(\Theta - 3)^2 + 1$:

$$\frac{d\mathcal{L}}{d\Theta} = 2(\Theta - 3)$$

This is a linear function of $\Theta$: it is zero at $\Theta^* = 3$ (the unique minimizer), negative for $\Theta < 3$ (gradient points left, so descent goes right), and positive for $\Theta > 3$ (descent goes left). The function is **strictly convex** (second derivative $= 2 > 0$ everywhere), so the single stationary point is the unique global minimum.

---

### Questions 4 & 5 — Comment on convergence, oscillations, and divergence for the three step sizes. Relate observations to theory.

**Theoretical background:**

Gradient Descent applies the update rule:
$$\Theta^{(k+1)} = \Theta^{(k)} - \eta \cdot \nabla\mathcal{L}(\Theta^{(k)}) = \Theta^{(k)} - 2\eta(\Theta^{(k)} - 3)$$

Defining the error $e^{(k)} = \Theta^{(k)} - 3$, the update becomes:
$$e^{(k+1)} = (1 - 2\eta)\, e^{(k)}$$

This is a geometric sequence with ratio $\rho = 1 - 2\eta$. Convergence requires $|\rho| < 1$, i.e. $0 < \eta < 1$.

For a general $L$-smooth function (Lipschitz gradient with constant $L$), convergence of GD with constant step size is guaranteed when $\eta \leq 1/L$. For $\mathcal{L}(\Theta) = (\Theta-3)^2 + 1$, the Lipschitz constant of the gradient is $L = 2$, so the safe range is $\eta \leq 0.5$.

**Analysis by step size:**

**$\eta = 0.05$ (too small):**
The contraction ratio is $\rho = 1 - 2(0.05) = 0.9$. Each step reduces the error by only 10%. Convergence is guaranteed but very slow — it takes on the order of $\log(\varepsilon)/\log(0.9) \approx 22 \cdot \log(1/\varepsilon)$ iterations to reach tolerance $\varepsilon$. The iterate sequence $\Theta^{(k)}$ creeps monotonically toward $\Theta^* = 3$, and the loss curve decreases smoothly but slowly. In high-dimensional or large-dataset problems, this translates into a prohibitive computational cost.

**$\eta = 0.2$ (well-chosen / "just right"):**
The ratio is $\rho = 1 - 2(0.2) = 0.6$. The error contracts by 40% at each step, giving fast monotone convergence. The iterate jumps steadily toward $\Theta^* = 3$ without overshooting. The loss curve shows a clean exponential decay. This is the regime recommended by theory: large enough for fast progress, small enough to avoid oscillations.

**$\eta = 1.0$ (too large / on the boundary of divergence):**
The ratio is $\rho = 1 - 2(1.0) = -1$. The iterate bounces symmetrically around $\Theta^* = 3$ without ever converging: if $\Theta^{(0)} = 0$, then $\Theta^{(1)} = 6$, $\Theta^{(2)} = 0$, and so on. The loss $\mathcal{L}(\Theta^{(k)})$ never decreases. For $\eta > 1$, the ratio $|\rho| > 1$ and the iterates diverge to infinity. This demonstrates why the step size must respect the smoothness of the loss landscape.

**Role of convexity:**

Since $\mathcal{L}$ is **strictly convex**, every stationary point is the unique global minimum. This means:
1. GD is guaranteed to converge to $\Theta^* = 3$ from *any* starting point, as long as $\eta$ is in the valid range — there are no local minima or saddle points to get trapped in.
2. The choice of $\Theta^{(0)}$ does not matter for the quality of the solution (only for the speed of convergence if initialized far from $\Theta^*$).

In contrast, for non-convex functions (as in Exercise 2), a stationary point may be a saddle or a local minimum, and the choice of starting point becomes critical.

---

## Exercise 2: Backtracking Line Search

$$\mathcal{L}(\Theta) = \Theta^4 - 3\Theta^2 + 2$$

---

### Question 4 — Discuss: (a) why different initializations converge to different minima; (b) how backtracking automatically chooses a suitable step size; (c) situations where constant step size would fail.

**Theoretical background:**

The gradient is $\nabla\mathcal{L}(\Theta) = 4\Theta^3 - 6\Theta$. Setting it to zero:
$$4\Theta^3 - 6\Theta = 0 \implies \Theta(4\Theta^2 - 6) = 0 \implies \Theta \in \left\{0,\; +\sqrt{3/2},\; -\sqrt{3/2}\right\}$$

The second derivative $\mathcal{L}''(\Theta) = 12\Theta^2 - 6$ tells us:
- $\Theta = 0$: $\mathcal{L}''(0) = -6 < 0$ → **local maximum** (saddle in 1D)
- $\Theta = \pm\sqrt{3/2} \approx \pm 1.225$: $\mathcal{L}''(\pm\sqrt{3/2}) = 18 - 6 = 12 > 0$ → **local minima**, symmetric with equal function value $\mathcal{L}(\pm\sqrt{3/2}) = 9/4 - 9/2 + 2 = -1/4$

**(a) Why different initializations converge to different minima:**

GD is a **local** algorithm: at each step it moves in the direction of steepest local descent, following the gradient at the current point. It has no global view of the landscape. The **basin of attraction** of a local minimum is the set of initial points from which GD converges to that minimum.

- $\Theta_0 = -2$: the gradient at $\Theta=-2$ is $4(-8) - 6(-2) = -32 + 12 = -20 < 0$, so GD moves in the positive direction, eventually entering the basin of $\Theta^* \approx -1.225$.
- $\Theta_0 = 0.5$: the gradient is $4(0.125) - 6(0.5) = 0.5 - 3 = -2.5 < 0$, pushing rightward toward $\Theta^* \approx +1.225$.
- $\Theta_0 = 2$: the gradient is $4(8) - 6(2) = 32 - 12 = 20 > 0$, GD moves leftward toward $\Theta^* \approx +1.225$.

The non-convexity of $\mathcal{L}$ means GD converges to the nearest local minimum in the direction of the local gradient field, not necessarily the global minimum. This is a fundamental limitation of first-order methods on non-convex landscapes.

**(b) How backtracking automatically chooses a suitable step size:**

The **Armijo backtracking** algorithm starts with an initial step size $\bar{\eta}$ and repeatedly shrinks it by a factor $\beta \in (0,1)$ until the **Armijo (sufficient decrease) condition** is satisfied:

$$\mathcal{L}(\Theta^{(k)} - \eta\, g^{(k)}) \leq \mathcal{L}(\Theta^{(k)}) - c\,\eta\,\|g^{(k)}\|^2, \qquad c \in (0,1),\; g^{(k)} = \nabla\mathcal{L}(\Theta^{(k)})$$

The right-hand side is a linear model of how much the loss *should* decrease if the function were linear. The condition enforces that the actual decrease is at least a fraction $c$ of this predicted decrease. This guarantees:
- **Monotone decrease**: the loss always goes down by a sufficient amount at each step.
- **Adaptive sizing**: in flat regions (small gradient norm) or near minima, a large step is accepted quickly; in regions with large curvature (large second derivative), the step is reduced automatically to avoid overshooting.
- **Termination guarantee**: for any smooth $\mathcal{L}$, the backtracking loop terminates in a finite number of shrinks, because for sufficiently small $\eta$ the Taylor expansion guarantees the condition holds.

For $\mathcal{L}(\Theta) = \Theta^4 - 3\Theta^2 + 2$, the curvature ($\mathcal{L}'' = 12\Theta^2 - 6$) varies strongly across the domain. Backtracking adapts: far from a minimum where curvature is high, it uses small steps; near the minimum where the function is smoother, it accepts larger steps and converges faster.

**(c) Situations where constant step size fails:**

1. **Varying curvature (non-convex or non-quadratic functions):** A constant $\eta$ valid near a minimum may be too large in a high-curvature region elsewhere, causing divergence. Conversely, a step small enough for the worst-case curvature wastes iterations in smoother regions.

2. **Multiple local minima near saddles:** For $\Theta_0 = 0.5$, a poorly tuned large constant step size could cause the iterate to jump over the local minimum at $\approx +1.225$ and land on the other side, potentially entering the basin of $-1.225$ or oscillating indefinitely.

3. **Flat regions followed by steep regions (plateaus):** With a constant step, the algorithm crawls through flat areas and risks overshooting when it reaches steep descent.

4. **Unknown Lipschitz constant:** A safe constant step size requires knowing the global Lipschitz constant $L$ of the gradient ($\eta \leq 1/L$). For general non-convex functions this constant is hard to estimate and may be very conservative.

---

## Exercise 3: GD in 2D

$$\mathcal{L}(\Theta) = \frac{1}{2}\Theta^T A\,\Theta, \quad A = \begin{bmatrix}1 & 0 \\ 0 & 25\end{bmatrix}$$

---

### Question 3 — Comment on: elongated ellipses, gradient direction vs level sets, zig-zag behaviour, slow convergence in narrow valleys.

**Theoretical background:**

For the quadratic $\mathcal{L}(\Theta) = \frac{1}{2}\Theta^T A\Theta$ with $A = \text{diag}(\lambda_1, \lambda_2)$, the gradient is $\nabla\mathcal{L}(\Theta) = A\Theta$.

The **level sets** $\{\Theta : \mathcal{L}(\Theta) = c\}$ are ellipses with semi-axes $\sqrt{2c/\lambda_1}$ and $\sqrt{2c/\lambda_2}$. With $\lambda_1 = 1$ and $\lambda_2 = 25$, the axis along $\Theta_2$ is $\sqrt{25} = 5$ times shorter than the axis along $\Theta_1$.

The **condition number** is $\kappa(A) = \lambda_{\max}/\lambda_{\min} = 25/1 = 25$.

**Elongated ellipses from ill-conditioning:**

The level sets are elongated in the direction of the *smallest* eigenvalue ($\Theta_1$-axis, $\lambda_1 = 1$) and compressed in the direction of the *largest* eigenvalue ($\Theta_2$-axis, $\lambda_2 = 25$). Geometrically, this creates a narrow valley: moving a unit distance along $\Theta_2$ raises the loss 25 times more than moving the same distance along $\Theta_1$.

This is the essence of **ill-conditioning**: the loss landscape is highly anisotropic, with very different curvatures in different directions. The condition number $\kappa = 25$ quantifies this anisotropy.

**Gradient direction vs level set lines:**

The gradient $\nabla\mathcal{L}(\Theta) = A\Theta = (\Theta_1,\, 25\Theta_2)$ is **not perpendicular to the level sets** in the Euclidean sense when $A \neq \lambda I$. More precisely, the gradient points perpendicular to level sets in the metric defined by $A$, but in the standard Euclidean metric it is "pulled" strongly toward the $\Theta_2$ direction due to the large eigenvalue. As a result, GD steps are dominated by the steep $\Theta_2$ component, even when most of the remaining distance to the minimum lies along the shallow $\Theta_1$ direction.

**Zig-zag behaviour for large condition numbers:**

The GD update is $\Theta^{(k+1)} = \Theta^{(k)} - \eta A\Theta^{(k)} = (I - \eta A)\Theta^{(k)}$.

The convergence rate in each eigendirection is governed by $|1 - \eta\lambda_i|$. For $\eta = 0.05$:
- $\Theta_1$-direction: $|1 - 0.05 \cdot 1| = 0.95$ → slow reduction (5% per step)
- $\Theta_2$-direction: $|1 - 0.05 \cdot 25| = 0.25$ → fast reduction (75% per step)

For $\eta = 0.05$ (borderline): $|1 - 0.05 \cdot 25| = 0.25 < 1$ → both directions converge, but $\Theta_1$ lags behind.

For $\eta = 0.1$: $|1 - 0.1 \cdot 25| = 1.5 > 1$ → the $\Theta_2$-component **diverges**, even though $\Theta_1$ would still converge. The maximum stable step size is $\eta < 2/\lambda_{\max} = 2/25 = 0.08$.

The zig-zag pattern emerges because to progress along the shallow $\Theta_1$-axis, $\eta$ must be large enough. But any $\eta$ large enough to make meaningful progress along $\Theta_1$ will cause oscillations or divergence along the steep $\Theta_2$-axis. Successive gradients alternate sign in the $\Theta_2$-direction, creating the characteristic zigzag path.

**Slow convergence in narrow valleys:**

The convergence rate of GD on a quadratic is bounded by:
$$\frac{\mathcal{L}(\Theta^{(k+1)})}{\mathcal{L}(\Theta^{(k)})} \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^2$$

For $\kappa = 25$: $\left(\frac{24}{26}\right)^2 \approx 0.852$ — each iteration reduces the loss by at most about 15%, requiring roughly $\log(1/\varepsilon)/\log(1/0.852) \approx 6.4\log(1/\varepsilon)$ iterations. For $\kappa = 1$ (perfectly conditioned), convergence would be exact in one step. This demonstrates why preconditioning (transforming the problem to reduce $\kappa$) or adaptive methods (like Adam) are critical in practice.

---

## Exercise 4: Exact Line Search vs Backtracking

$$\mathcal{L}(\Theta) = \frac{1}{2}\Theta^T A\Theta, \quad A = \begin{bmatrix}5 & 0 \\ 0 & 2\end{bmatrix}$$

---

### Question 4 — Compare speed of convergence and smoothness of step sizes.

**Theoretical background:**

For a quadratic $\mathcal{L}(\Theta) = \frac{1}{2}\Theta^T A\Theta$, the **exact line search** (also called steepest descent with optimal step) computes the analytically optimal step size at each iteration:

$$\eta_k^* = \frac{g_k^T g_k}{g_k^T A\, g_k}, \qquad g_k = A\Theta^{(k)}$$

This is derived by minimizing $\mathcal{L}(\Theta^{(k)} - \eta g_k)$ over $\eta$ analytically, setting $\frac{d}{d\eta}\mathcal{L}(\Theta^{(k)} - \eta g_k) = 0$.

The condition number here is $\kappa(A) = 5/2 = 2.5$.

**Speed of convergence:**

Exact line search achieves the theoretical optimal convergence rate for steepest descent on a quadratic. The classical result is:

$$\mathcal{L}(\Theta^{(k+1)}) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^2 \mathcal{L}(\Theta^{(k)})$$

With $\kappa = 2.5$: $\left(\frac{1.5}{3.5}\right)^2 \approx 0.184$. Each step reduces the loss by at least 81.6% — very fast convergence given the mild conditioning.

Backtracking does not compute the analytical optimum; it finds the first $\eta$ satisfying the Armijo condition. Its step sizes are generally suboptimal (smaller than $\eta_k^*$ due to the conservative shrinking), leading to slightly more iterations. However, the difference is modest for well-conditioned problems and the overhead per iteration is similar.

**Smoothness of step sizes:**

- **Exact line search:** $\eta_k^*$ varies smoothly iteration to iteration, following the geometry of the quadratic exactly. For a pure quadratic, it can be shown that successive gradients are orthogonal ($g_{k+1}^T g_k = 0$), and $\eta_k^*$ oscillates between two values determined by $\lambda_{\min}$ and $\lambda_{\max}$.

- **Backtracking:** the step size at each iteration depends on the initial guess $\bar{\eta}$ and how many times the shrinkage factor $\beta$ is applied. It is generally noisier and less predictable, producing an irregular step-size sequence. For a quadratic, it may accept the same step for several iterations and then suddenly shrink, unlike the smooth variation of exact line search.

**Key takeaway:**

Exact line search is theoretically superior for quadratics (optimal step, provably fastest convergence per iteration for steepest descent), but it requires computing $g_k^T A g_k$, which needs explicit knowledge of $A$. For general non-quadratic or non-convex $\mathcal{L}$, no closed-form exists and backtracking becomes the practical alternative. On this moderately-conditioned problem ($\kappa = 2.5$), both methods converge quickly; the advantage of exact line search would be more visible for large $\kappa$.

---

## Exercise 5: Gradient Descent on the Rosenbrock Function

$$\mathcal{L}(\Theta) = (1 - \Theta_1)^2 + 100(\Theta_2 - \Theta_1^2)^2$$

Unique global minimum: $\Theta^* = (1, 1)$, $\mathcal{L}(\Theta^*) = 0$.

---

### Questions 3 & 4 — Visualize and analyze the trajectories. Discuss entering the valley, zig-zagging, step size issues.

**Theoretical background:**

The Rosenbrock function is the canonical benchmark for testing optimization algorithms on challenging non-quadratic, non-convex landscapes. Despite having a single global minimum, its geometry makes pure GD struggle.

**The gradient:**

$$\frac{\partial \mathcal{L}}{\partial \Theta_1} = -2(1 - \Theta_1) - 400\Theta_1(\Theta_2 - \Theta_1^2)$$
$$\frac{\partial \mathcal{L}}{\partial \Theta_2} = 200(\Theta_2 - \Theta_1^2)$$

**The Hessian (local curvature):**

$$H(\Theta) = \begin{bmatrix} 2 + 1200\Theta_1^2 - 400\Theta_2 & -400\Theta_1 \\ -400\Theta_1 & 200 \end{bmatrix}$$

At the minimum $(1,1)$: $H = \begin{bmatrix} 802 & -400 \\ -400 & 200 \end{bmatrix}$, with eigenvalues approximately $\lambda_{\min} \approx 0.47$ and $\lambda_{\max} \approx 1001.5$, giving a condition number $\kappa \approx 2500$. This extreme ill-conditioning is the root cause of all GD difficulties on Rosenbrock.

**Why the landscape is challenging:**

1. **The narrow curved valley:** The minimum lies along the parabolic curve $\Theta_2 = \Theta_1^2$. The function is extremely flat along this curve (curvature $\approx \lambda_{\min}$) but extremely steep perpendicular to it (curvature $\approx \lambda_{\max}$). This creates a ratio of 2500:1 between directions.

2. **The condition number varies across the domain:** Away from the minimum, the curvature is even more extreme, making the effective $L$ (Lipschitz constant of the gradient) very large in parts of the domain.

**Analysis by initialization and method:**

**$\Theta^{(0)} = (-1.5, 2)$ and $\Theta^{(0)} = (-1, 0)$ (outside the valley, steep region):**
Starting far from the valley, the gradient is dominated by the $(\Theta_2 - \Theta_1^2)$ term. GD can make rapid initial progress in the $\Theta_2$-direction (descending toward the valley floor), but once it approaches the valley it encounters the severe ill-conditioning. The gradient then points nearly perpendicularly to the valley direction, causing the iterates to oscillate across the valley without progressing along it toward $(1,1)$.

**$\Theta^{(0)} = (0, 2)$ (above the valley):**
The descent to the valley floor is fast, but entering the valley and then navigating along it toward the minimum is slow due to zig-zagging.

**$\Theta^{(0)} = (1.5, 1.5)$ (close to the minimum):**
GD enters the narrow valley quickly, but the severe local curvature near $(1,1)$ still causes zig-zagging. This initialization shows that proximity to the minimum does not guarantee fast convergence with pure GD.

**Constant step size $\eta = 10^{-3}, 10^{-4}, 10^{-5}$:**

The Lipschitz constant $L$ of $\nabla\mathcal{L}$ depends on the maximum curvature. Near the minimum $L \approx \lambda_{\max}(H) \approx 1001$, so the safe step size is $\eta \leq 1/L \approx 10^{-3}$.

- $\eta = 10^{-3}$: the only constant step size that converges in 10,000 iterations. After 10k steps it reaches $\mathcal{L} \approx 10^{-4}$ to $10^{-5}$ depending on initialization — close to the minimum but not there. This is the borderline-stable regime: the step is at the edge of the safe bound, producing heavy zig-zagging across the valley while still making net progress along it.

- $\eta = 10^{-4}$: fails to converge from all initializations in 10k steps. Final losses range from $3.8 \times 10^{-2}$ (from the easiest start $(1.5, 1.5)$) to $2.78$ (from the hardest $(-1.5, 2)$). The step is 10× too small to make meaningful progress along the shallow valley direction in a fixed budget of iterations — the iterates are still clearly far from $(1,1)$.

- $\eta = 10^{-5}$: essentially frozen. Final losses are close to the *initial* loss values ($5.6$, $1.9$, $0.5$, $0.06$), confirming that 10,000 steps at this step size produce negligible displacement. This is a practical illustration of the "too small → stuck" failure mode.

In all cases with constant $\eta$, the zig-zag pattern is visible: GD oscillates across the valley width while barely advancing along its length, because the step required to advance along the flat direction ($\sim 1/\lambda_{\min}$) is catastrophically large relative to the stable step for the steep direction ($\sim 1/\lambda_{\max}$). The numerical results confirm that only $\eta = 10^{-3}$ sits in the narrow usable range, and even then 10k iterations are not sufficient to reach high precision.

**Backtracking line search:**

Backtracking adapts the step size at each iteration. When GD is crossing the valley (steep curvature), it automatically reduces $\eta$; when it is aligned with the shallow valley direction, it accepts larger steps. The numerical results confirm this clearly: after 10,000 iterations backtracking reaches $\mathcal{L} \approx 10^{-7}$ to $10^{-10}$ from all four initializations — 2 to 5 orders of magnitude better than $\eta = 10^{-3}$, and the result is robust regardless of starting point.

However, even backtracking cannot fully overcome the fundamental ill-conditioning. The Armijo condition only guarantees *sufficient decrease*, not *optimal progress along the valley*. The method still exhibits slow convergence on Rosenbrock because the gradient direction is nearly orthogonal to the path toward the minimum when inside the curved valley. The fact that 10,000 iterations are still needed to reach $\mathcal{L} \sim 10^{-9}$ rather than machine precision confirms this — a well-conditioned problem would converge in far fewer steps.

**Key conclusion — limitations of pure Gradient Descent:**

The Rosenbrock function demonstrates all the fundamental limitations of first-order gradient descent:
1. **Ill-conditioning ($\kappa \approx 2500$):** the convergence bound $\left(\frac{\kappa-1}{\kappa+1}\right)^2 \approx 0.998$ implies that each step reduces the loss by at most 0.2%, requiring on the order of $10^3$–$10^4$ iterations for modest accuracy.
2. **Non-quadratic curvature:** $\kappa$ is not constant across the domain, so no fixed $\eta$ is globally optimal.
3. **Misaligned gradient:** inside the curved valley, $\nabla\mathcal{L}$ points almost perpendicular to the direction of maximum progress, making GD systematically inefficient.

These are precisely the limitations that motivate second-order methods (Newton's method, which uses the Hessian to pre-condition the gradient direction) and adaptive first-order methods (Adam, RMSProp), which approximate second-order information to handle ill-conditioning without explicit Hessian computation.

---

## Summary of Key Theoretical Takeaways

| Concept | Core Idea |
|---|---|
| **GD update rule** | $\Theta^{(k+1)} = \Theta^{(k)} - \eta_k \nabla\mathcal{L}(\Theta^{(k)})$ — move opposite to gradient |
| **Convexity** | Unique global min; every stationary point is the minimum; initialization irrelevant for quality |
| **Step size (constant)** | Too small → slow; too large → oscillations/divergence; safe range $\eta < 2/L$ |
| **Backtracking (Armijo)** | Adaptive $\eta$: shrink until sufficient decrease; handles varying curvature automatically |
| **Condition number $\kappa$** | $\kappa = \lambda_{\max}/\lambda_{\min}$; large $\kappa$ → elongated level sets, zig-zag, slow convergence |
| **Non-convexity** | Multiple stationary points; GD converges to a local min determined by initialization |
| **Rosenbrock** | Extreme $\kappa \approx 2500$; curved narrow valley; gradient nearly orthogonal to valley → GD fails in practice |
| **Motivation for SGD/Adam** | GD's limitations (cost, sensitivity, ill-conditioning) drive stochastic and adaptive methods |
