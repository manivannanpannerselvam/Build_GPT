"""Level 1 - Concept 1: What does "learning" mean?

Pure Python, no libraries. Run:  python3 01_what_is_learning.py

Task: learn the hidden rule  price = 2 * size  from examples.
"""

# 1. DATA: inputs (size) with the correct answers (price).
#    The model never sees the rule, only these examples.
data = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]  # (x, true_y)

# 2. PARAMETERS (weights): the adjustable knobs. Start with a bad guess.
w = 0.5


# 3. MODEL: a function that uses the parameters to turn input into output.
def model(x, w):
    return w * x


# 4. PREDICTION: what the model outputs for one input.
print("Before learning:")
print(f"  w = {w}, model(10) = {model(10, w)}   (true answer is 20)")


# 5. LOSS: ONE number saying how wrong the model is over all the data.
#    Mean squared error = average of (prediction - truth)^2. 0 means perfect.
def loss_fn(w):
    total = 0
    for x, true_y in data:
        error = model(x, w) - true_y
        total += error ** 2
    return total / len(data)


print(f"  loss = {loss_fn(w):.3f}  (high = bad)\n")

# 6. LEARNING: adjust the parameter to reduce the loss.
#    Simple idea: try nudging w up and down, keep whichever lowers the loss.
#    (Real models use gradients to pick the direction. That is Concept 5.)
step = 0.1
print("Learning:")
for i in range(1, 31):
    current = loss_fn(w)
    if loss_fn(w + step) < current:
        w = w + step      # nudging up helped
    elif loss_fn(w - step) < current:
        w = w - step      # nudging down helped
    # else: neither helped, so we're as good as this step size allows
    if i % 5 == 0 or i == 1:
        print(f"  step {i:2d}: w = {w:.2f}, loss = {loss_fn(w):.3f}")

# 7. RESULT: the model "discovered" the rule from data alone.
print("\nAfter learning:")
print(f"  w = {w:.2f}  (true rule uses 2)")
print(f"  model(10) = {model(10, w):.2f}  (true answer is 20)")
