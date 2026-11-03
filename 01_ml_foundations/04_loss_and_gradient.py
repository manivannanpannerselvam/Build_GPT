"""Level 1 - Concept 4: Loss and gradient (gradient descent)

Pure Python, no libraries. Run:  python3 04_loss_and_gradient.py

Task: learn  price = w * size.  The true rule is w = 2.
Loss tells us HOW wrong we are. Gradient tells us WHICH WAY to move w.
"""

# sizes = [1, 2, 3, 4, 5]
# prices = [2, 4, 6, 8, 10]

sizes = [1, 2, 3, 4, 5]
prices = [2, 4, 10, 15, 20]


def loss(w):
    """Mean squared error of the model price = w * size."""
    total = 0
    for s, p in zip(sizes, prices):
        total += (w * s - p) ** 2
    return total / len(sizes)


# 1. LOSS IS A HILL (really a bowl). Each w has a height = its loss.
print("1. Loss for different w (true w = 2)")
for w in [0, 1, 2, 3, 4]:
    bar = "#" * int(loss(w))
    print(f"  w={w}  loss={loss(w):>6.2f}  {bar}")
print("  Bowl shape: lowest at w=2. Goal: walk down to the bottom.\n")

# 2. GRADIENT = SLOPE of the loss at the current w.
#    Measure it by brute force: nudge w a tiny bit and see how loss changes.
#      slope = (loss(w + tiny) - loss(w)) / tiny


def numeric_gradient(w, tiny=0.0001):
    return (loss(w + tiny) - loss(w)) / tiny


print("2. Gradient = slope (measured by nudging w a tiny bit)")
for w in [0, 1, 2, 3, 4]:
    g = numeric_gradient(w)
    if g < -0.01:
        advice = "negative slope -> increase w"
    elif g > 0.01:
        advice = "positive slope -> decrease w"
    else:
        advice = "flat -> we are at the bottom"
    print(f"  w={w}  gradient={g:>8.2f}  {advice}")
print("  The sign tells the direction. The size tells how steep.\n")


# 3. THE FORMULA FOR THE SLOPE (from calculus, no need to nudge)
#      loss     = average of (w*s - p)^2
#      gradient = average of 2 * (w*s - p) * s
def gradient(w):
    total = 0
    for s, p in zip(sizes, prices):
        error = w * s - p          # prediction minus truth
        total += 2 * error * s     # how much this example pushes w
    return total / len(sizes)


print("3. Formula matches the nudge measurement")
for w in [0, 1, 3]:
    print(f"  w={w}  formula={gradient(w):>7.2f}  nudged={numeric_gradient(w):>7.2f}")
print()

# 4. ONE STEP BY HAND:  w_new = w - lr * gradient
#    Minus sign: walk AGAINST the slope, i.e. downhill.
lr = 0.05
w = 0.0
g = gradient(w)
print("4. One step by hand")
print(f"  start   w={w}  loss={loss(w):.2f}  gradient={g:.2f}")
w_new = w - lr * g
print(f"  update  w = {w} - {lr} * ({g:.2f}) = {w_new:.2f}")
print(f"  after   w={w_new:.2f}  loss={loss(w_new):.2f}   (loss went down)\n")

# 5. GRADIENT DESCENT = repeat that step many times
print("5. Gradient descent loop (lr=0.05)")
w = 0.0
for step in range(1, 21):
    w = w - lr * gradient(w)
    if step <= 5 or step % 5 == 0:
        print(f"  step {step:>2}  w={w:.4f}  loss={loss(w):.5f}")
print("  Steps shrink by themselves because the slope flattens near the bottom.\n")

# 6. LEARNING RATE = step size. Compare three choices, 10 steps each.
print("6. Learning rate matters (10 steps, start w=0)")
for lr, name in [(0.001, "too small"), (0.05, "just right"), (0.2, "too big")]:
    w = 0.0
    for _ in range(10):
        w = w - lr * gradient(w)
    print(f"  lr={lr:<5} ({name:<10})  w={w:>12.4f}  loss={loss(w):>14.4f}")
print("  too small: barely moved.  too big: overshoots the bottom and explodes.\n")

# 7. MORE THAN ONE WEIGHT: price = w * size + b
#    Each weight gets its OWN gradient. Update them all together.
#    (This is exactly what happens in GPT, just with billions of weights.)
sizes2 = [1, 2, 3, 4, 5]
prices2 = [5, 7, 9, 11, 13]      # true rule: price = 2*size + 3

w, b, lr = 0.0, 0.0, 0.02
for _ in range(2000):
    dw = db = 0
    for s, p in zip(sizes2, prices2):
        error = (w * s + b) - p
        dw += 2 * error * s / len(sizes2)    # slope for w
        db += 2 * error / len(sizes2)        # slope for b
    w -= lr * dw
    b -= lr * db
print("7. Two weights (true rule: w=2, b=3)")
print(f"  learned w={w:.3f}  b={b:.3f}")

# KEY IDEAS
#   - Loss: one number, how wrong the model is.
#   - Gradient: slope of the loss for each weight; says which way is uphill.
#   - Update: weight = weight - lr * gradient (step downhill).
#   - Learning rate: too small is slow, too big overshoots.
#   - Repeat until the loss stops going down. That is "training".


# Your understanding is mostly right. Two small corrections:

# 1. Gradient sign and loss. Your picture is correct.

# If w is below the answer (e.g. w=0), the gradient is negative. That means increase w.
# If w is above the answer (e.g. w=4), the gradient is positive. That means decrease w.
# At the answer (w=2), the gradient is 0 and the loss is 0.
# Loss can't be negative with MSE, because squared errors are never below 0. So the lowest loss is 0, not -0. The gradient is the one that goes negative → 0 → positive.

# 2. 0.05, 0.2 and 0.5 are learning rates, not weight increases. The weight changes by lr × gradient at each step. The learning rate is the knob you choose. The weight is what the model learns.

# For this problem the update shrinks the distance to the answer by a factor of (1 − 22 × lr) each step. That 22 comes from this particular data.

# lr	Factor	Result
# 0.01	0.78	Too small. It moves toward 2 but slowly.
# 0.05	−0.1	Just right. It reaches 2 in about 4 steps.
# 0.1	−1.2	Too big. It jumps past 2 and each jump is larger.
# 0.2	−3.4	Too big. It explodes quickly.
# 0.5	−10	Too big. It explodes even faster.




