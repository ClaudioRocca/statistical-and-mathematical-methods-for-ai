# Homework 3 – Supervised Learning: Theoretical Answers
**Author:** Claudio Rocca

---

## Exercise 1: Logistic Regression on a Toy 2D Dataset

**Experimental results (N=10000, lr=0.1, 1000 epochs of full GD):**

| Θ₀ (bias) | Θ₁ | Θ₂ | Final BCE Loss | Final Accuracy |
|---|---|---|---|---|
| -0.2889 | 1.9096 | 1.9385 | 0.006736 | 99.90% |

---

### Question 6 — Comment on why the decision boundary is linear.

**Why logistic regression produces a linear decision boundary:**

The model predicts class 1 when $\sigma(\Theta^T x) \geq 0.5$, which is equivalent to $\Theta^T x \geq 0$. The decision boundary is therefore the set of points satisfying:
$$\Theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$$

This is a linear equation in the input features — geometrically a hyperplane (a line in 2D). Logistic regression is a **linear classifier**: it can only separate classes with a hyperplane, regardless of the learning algorithm used to train it. The sigmoid function introduces nonlinearity in the output probability, but the decision rule itself is determined by the sign of the linear score $\Theta^T x$.

For this toy dataset the two Gaussian clusters are well separated (centers at $(-2,-2)$ and $(2,2)$), so a linear boundary is sufficient to achieve near-perfect accuracy. The learned parameters confirm this: $\theta_1 \approx \theta_2 \approx 1.91$, meaning the boundary is roughly diagonal ($x_1 + x_2 = 0.29$), which geometrically bisects the segment between the two cluster centers — exactly what the optimal Bayes decision boundary would do for equal-covariance Gaussians.

The 0.10% error rate comes from the overlap region of the two Gaussian distributions, which is irreducible regardless of the model — it is **Bayes error**. No linear (or nonlinear) model can do better on this data.

---

## Exercise 2: SGD on Logistic Regression

**Experimental results (100 epochs, lr=0.1):**

| Batch size | Final Loss | Final Accuracy | Loss std (epoch-to-epoch Δ) |
|---|---|---|---|
| 1 | 0.0017 | 99.94% | 0.06875 |
| 10 | 0.0017 | 99.94% | 0.06849 |
| N=10000 (≡ GD) | 0.0171 | 99.89% | 0.03191 |

---

### Questions 3 & 4 — Compare batch sizes; explain noise in small batches and smoothness in large batches.

**Convergence speed and quality:**

Small batches (1 and 10) converge to a significantly lower final loss (0.0017) than full-batch GD (0.0171) after the same number of epochs. The reason is identical to what was observed in HW2: with batch=1 and N=10000, each epoch performs 10,000 parameter updates instead of 1. In 100 epochs that is 1,000,000 updates vs 100 — the small-batch methods have taken far more gradient steps and arrived much closer to the optimum.

**Why gradients are noisier for small batches:**

The stochastic gradient computed on a mini-batch of size $b$ is:
$$g_k = \frac{1}{b}\sum_{i \in \mathcal{M}_k} \nabla_\Theta \ell(\Theta; x^{(i)}, y^{(i)})$$

This is an unbiased estimator of the true gradient, but its variance is $\text{Var}(g_k) \propto 1/b$. For $b=1$ a single misclassified or borderline sample can produce a gradient that points in a very different direction from the full-dataset gradient. This causes the loss curve to fluctuate strongly from epoch to epoch.

The `loss_std` metric (standard deviation of epoch-to-epoch loss changes) quantifies this: batch=1 gives 0.069 and batch=10 gives 0.068, both much larger than full-batch GD at 0.032. Full GD is fully deterministic, so its only source of variation is the monotone decrease due to gradient steps — hence the smaller std.

**Why larger batches give smoother curves:**

With $b = N$, the gradient is exact and identical at every epoch given the same $\Theta$. The loss decreases monotonically and the accuracy climbs smoothly. There are no fluctuations because no randomness enters the computation. The tradeoff is a far slower approach to the optimum per epoch, as each epoch counts as only one gradient step.

---

## Exercise 3: Evaluation Metrics on a Synthetic Dataset

**Results using GD-trained model (Θ̂ from Exercise 1):**

| Threshold | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 10000 | 40 | 0 | 9960 | 0.9980 | 0.9960 | **1.0000** | 0.9980 |
| 0.5 | 9999 | 19 | 1 | 9981 | 0.9990 | 0.9981 | 0.9999 | 0.9990 |
| 0.7 | 9990 | 3 | 10 | 9997 | 0.9993 | **0.9997** | 0.9990 | 0.9993 |

---

### Question 5 — Comment on how thresholds affect precision and recall, and why metrics depend on the application.

**How lower thresholds increase recall and lower precision:**

The threshold $\tau$ converts probabilities into hard predictions: $\hat{y} = 1$ if $\sigma(\Theta^T x) \geq \tau$. Lowering $\tau$ means the model predicts positive more readily — it classifies borderline samples as positive.

- **Recall** = TP / (TP + FN): the number of missed positives (FN) decreases because the model is less strict about who it calls positive. At $\tau=0.3$, recall is exactly 1.0000 — every true positive is caught, FN=0.
- **Precision** = TP / (TP + FP): the number of false alarms (FP) increases because borderline negatives now get classified as positive. At $\tau=0.3$, FP=40 vs FP=19 at $\tau=0.5$.

**How higher thresholds increase precision and reduce recall:**

Raising $\tau$ means the model only predicts positive when it is highly confident. Borderline positives now get classified as negative, so:
- FP drops (fewer false alarms → higher precision: 0.9997 at $\tau=0.7$),
- FN rises (more missed positives → lower recall: 10 FN at $\tau=0.7$ vs 0 at $\tau=0.3$).

**Why the choice of threshold depends on the application:**

There is a fundamental precision–recall tradeoff. The right threshold is not universally 0.5 — it depends on the relative cost of each type of error:

- **Medical screening (e.g., cancer or diabetes):** missing a true positive (FN) is far more dangerous than a false alarm (FP). A low threshold ($\tau=0.3$) maximises recall at the cost of more unnecessary follow-up tests — an acceptable tradeoff.
- **Spam detection:** falsely flagging a legitimate email as spam (FP) is worse than missing spam (FN). A high threshold ($\tau=0.7$) is preferable.
- **Fraud detection:** FP (blocking a valid transaction) has business cost; FN (missing fraud) has financial and reputational cost. The threshold should be calibrated to the organisation's loss function.

The F1-score provides a single balanced summary, but it weights precision and recall equally — which is only appropriate when their costs are equal. In the results, F1 is highest at $\tau=0.7$ (0.9993) because on this near-perfectly-separated dataset the model is so confident that the gain in precision at high threshold slightly outweighs the small loss in recall.

---

## Exercise 4: Logistic Regression on the Diabetes Dataset

**Dataset:** 768 samples, 8 features, 34.9% positive (class imbalance). SGD: batch=32, lr=1e-3, 200 epochs. Adam: batch=32, lr=1e-3, 200 epochs.

**Final results:**

| Method | Final BCE Loss | Final Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| SGD | 0.4753 | 77.73% | 0.6964 | 0.6418 | 0.6680 | 172 | 75 | 96 | 425 |
| Adam | 0.4484 | 77.60% | 0.7143 | 0.5970 | 0.6504 | 160 | 64 | 108 | 436 |

---

### Question 1d — Explain why normalisation is required.

**Conditioning:** Without normalisation, features such as Glucose (range ~0–200) and DiabetesPedigreeFunction (range ~0–2.5) differ by two orders of magnitude. The loss landscape becomes extremely elongated (large condition number), causing slow zig-zagging convergence identical to the ill-conditioning problem studied in HW1. Normalising to mean 0 and variance 1 makes all features contribute equally to the gradient, producing more circular loss contours and faster convergence.

**Stable optimisation:** With unnormalised features, the gradient magnitude is dominated by large-scale features. A learning rate safe for large features will be too small for small ones, and vice versa. Normalisation ensures that all parameter updates are on the same scale, allowing a single learning rate to work well for all dimensions.

**Meaningful gradient magnitudes:** The gradient of the BCE loss with respect to $\Theta_j$ is proportional to the feature values $x_j$. Without normalisation, the magnitude of $\nabla_{\Theta_j}\mathcal{L}$ depends on the units of feature $j$, making it impossible to compare or interpret the learned weights meaningfully.

### Question 2c — Comment on SGD loss and accuracy curves.

The SGD loss curve decreases steadily from ~0.63 (random initialisation, binary classification baseline) toward ~0.475, and accuracy climbs from ~65% to ~77.7%. Both curves are smooth by epoch 50 onwards, indicating that with batch=32 out of N=768, there are $768/32 = 24$ gradient steps per epoch — enough averaging to suppress most noise. The convergence is not complete at 200 epochs (the loss has not plateaued to a flat line), suggesting more epochs or a higher learning rate would further improve results.

### Question 5 — Which method converges faster, which oscillates more, and why (adaptive learning rates).

**Speed of convergence — Adam is faster:**

Adam reaches a lower final loss (0.448 vs 0.475) in the same 200 epochs, confirming faster convergence. In the early epochs (roughly epochs 1–50), Adam's loss curve drops more steeply than SGD's. This is due to **adaptive per-parameter learning rates**: Adam maintains exponential moving averages of the gradient ($\hat{m}_k$) and its square ($\hat{v}_k$), and scales each parameter update by $1/\sqrt{\hat{v}_k}$. Parameters with consistently large gradients get a smaller effective step, while those with small or sparse gradients get a larger one. This automatic rescaling achieves something similar to preconditioning the gradient by an approximation of the curvature, without requiring the full Hessian.

**Oscillation — SGD oscillates more:**

Plain SGD applies the same learning rate $\eta$ to every parameter at every step. When the gradient is noisy (as with batch=32) or the loss surface has different curvatures in different directions (which is typical for real datasets), SGD oscillates around the optimum rather than converging smoothly. Adam's momentum term ($\hat{m}_k$, the running average of the gradient) damps oscillations: it smooths out random fluctuations and reinforces consistent directions, producing a more stable descent trajectory.

**Adam's precision vs SGD's recall:**

Interestingly, Adam achieves higher precision (0.714 vs 0.696) but lower recall (0.597 vs 0.642) than SGD. Adam's smaller loss suggests it found a sharper minimum with a more conservative decision boundary — it is more careful about predicting positive, hence fewer FP (64 vs 75) but more FN (108 vs 96). Whether this is better or worse depends on the application: for diabetes screening, SGD's higher recall (fewer missed diabetic patients) is clinically preferable, even at the cost of more false alarms.

**Why Adam does not always win:**

Despite the lower loss, Adam's accuracy is marginally lower than SGD (77.60% vs 77.73%). The two methods are converging to different local minima of the non-convex (due to the neural network structure) or, in this linear case, to slightly different points in the flat basin of the convex BCE loss. The plateau at ~0.448 for Adam and ~0.475 for SGD both represent a similar classification boundary — the accuracy difference is within noise. For a linear model on this dataset, the fundamental ceiling (~78%) is set by the problem difficulty (class overlap, non-linear structure) rather than the optimiser.

---

## Optional Extension: Neural Network vs Logistic Regression

**Results (200 epochs, lr=1e-3, batch=32, hidden=16):**

| Model | Final BCE Loss | Final Accuracy |
|---|---|---|
| Logistic Regression (SGD) | 0.4753 | 77.73% |
| Logistic Regression (Adam) | 0.4484 | 77.60% |
| Neural Network (SGD) | 0.4675 | 76.82% |

**Why the neural network does not outperform logistic regression here:**

The neural network — one hidden layer with 16 ReLU units — has higher capacity than logistic regression and can in principle learn non-linear decision boundaries. However, it performs slightly worse (76.82% vs 77.73%) in this experiment for several reasons:

1. **Small dataset (768 samples):** with more parameters to fit ($d \times 16 + 16 + 16 \times 1 + 1 = 153$ for $d=9$, vs $d+1=10$ for logistic regression), the network is more prone to overfitting or slow convergence on small data.
2. **Same hyperparameters:** the network uses lr=1e-3 designed for logistic regression. Neural networks typically benefit from higher learning rates or Adam optimisation, which was not applied here.
3. **Short training (200 epochs):** neural networks generally require more epochs than logistic regression to converge from random initialisations (He initialisation was used, which is correct, but the network still needs more steps).
4. **Problem may be approximately linear:** the diabetes prediction task with these 8 features may not have strong non-linear structure that a neural network can exploit. Logistic regression's linear boundary may already capture most of the predictable variance, and the extra expressivity of the network does not help.

This illustrates an important practical lesson: **model capacity alone does not guarantee better performance**. For small, approximately-linear datasets, simpler models generalise better because they have less variance. The neural network's advantage appears on larger datasets with genuine non-linear structure (e.g., image classification), where logistic regression's linear boundary is provably insufficient.

---

## Summary of Key Theoretical Takeaways

| Concept | Core Idea |
|---|---|
| **Logistic regression** | Linear score → sigmoid → probability; decision boundary is always a hyperplane |
| **BCE loss** | Derived from maximum likelihood; penalises confident wrong predictions more than MSE |
| **Gradient of BCE** | $\nabla_\Theta\mathcal{L} = X^T(\hat{y} - y)/N$ — residual times input, same form as linear regression |
| **Linear classifier limit** | Cannot separate non-linearly separable data; neural networks overcome this with nonlinear activations |
| **Precision–recall tradeoff** | Lowering threshold ↑ recall ↓ precision; choice depends on relative cost of FP vs FN |
| **Adam vs SGD** | Adam adapts per-parameter step sizes via gradient moments; faster convergence, less oscillation |
| **Normalisation** | Required for conditioning, stable gradients, and equal feature scales |
| **Model capacity vs data size** | More capacity only helps when data is large enough and genuinely non-linear |
