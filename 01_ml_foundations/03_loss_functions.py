"""Level 1 - Concept 3: Loss functions

Pure Python, no libraries. Run:  python3 03_loss_functions.py

A loss function turns "how wrong is the model?" into ONE number.
Low loss = predictions close to the truth. Training = making loss smaller.
"""
import math

# 1. REGRESSION LOSS: predicting a number (house price)
#    true price vs. what the model predicted
y_true = [100, 50, 70]
y_pred = [80, 55, 70]

print("1. Regression: errors per example")
for t, p in zip(y_true, y_pred):
    print(f"  true={t:>3}  predicted={p:>3}  error={t - p:>3}  squared={(t - p) ** 2}")


def mse(y_true, y_pred):
    """Mean Squared Error: average of (error)^2.
    Squaring stops +/- errors cancelling and punishes big mistakes more."""
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def mae(y_true, y_pred):
    """Mean Absolute Error: average of |error|. Treats all errors evenly."""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


print(f"  MSE = {mse(y_true, y_pred):.2f}")
print(f"  MAE = {mae(y_true, y_pred):.2f}\n")

# 2. WHY SQUARE? One big miss hurts much more under MSE than MAE.
small_misses = [90, 40, 60]     # each off by 10
one_big_miss = [100, 50, 40]    # two perfect, one off by 30
print("2. MSE punishes big mistakes")
print(f"  3 misses of 10 -> MAE={mae(y_true, small_misses):.1f}  MSE={mse(y_true, small_misses):.1f}")
print(f"  1 miss of 30   -> MAE={mae(y_true, one_big_miss):.1f}  MSE={mse(y_true, one_big_miss):.1f}")
print("  Same MAE, but MSE is 3x bigger for the single large miss.\n")

# 3. CLASSIFICATION LOSS: predicting a category (spam or not)
#    Model outputs a probability. Truth: the email IS spam (label 1).
#    Cross-entropy = -log(probability given to the correct answer)


def cross_entropy(p_correct):
    return -math.log(p_correct)


print("3. Classification: cross-entropy (email really is spam)")
for p in [0.99, 0.60, 0.10, 0.01]:
    print(f"  model says spam with p={p:<4}  loss={cross_entropy(p):.2f}")
print("  Confident and right -> tiny loss. Confident and wrong -> huge loss.")
print("  GPT uses this too: loss = -log(prob it gave to the real next word).\n")

# 4. USING LOSS TO LEARN (same nudging idea as Concept 1)
#    Model: price = w * size. True rule: price = 2 * size.
sizes = [1, 2, 3, 4, 5]
prices = [2, 4, 6, 8, 10]


def loss_for(w):
    return mse(prices, [w * s for s in sizes])


print("4. Loss as a score for w (true w is 2)")
for w in [0, 1, 2, 3, 4]:
    print(f"  w={w}  loss={loss_for(w):>6.2f}")
print("  Loss is lowest at w=2. Training = searching for the w with lowest loss.")

w, step = 0.0, 0.1
for _ in range(50):
    if loss_for(w + step) < loss_for(w):
        w += step
    elif loss_for(w - step) < loss_for(w):
        w -= step
print(f"  after nudging: w={w:.2f}  loss={loss_for(w):.4f}")

# KEY IDEAS
#   - Loss = one number measuring wrongness; smaller is better.
#   - Regression (numbers): MSE or MAE.
#   - Classification (categories): cross-entropy.
#   - The loss you pick decides what the model considers a "bad" mistake.
