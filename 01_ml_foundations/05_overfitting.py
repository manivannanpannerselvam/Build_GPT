"""Level 1 - Concept 5: Overfitting, step by step

Pure Python, no libraries. Run:  python3 05_overfitting.py

Big idea: a model can score perfectly on data it has SEEN
and still be bad on data it has NOT seen. That is overfitting.
"""

# STEP 1. The data: (size, price). The real rule is roughly price = 2 * size,
#         but real data is messy, so prices are a bit off (this is "noise").
train = [(1, 3), (2, 3), (3, 8), (4, 7), (5, 12)]   # houses the model learns from
test = [(6, 11), (7, 15), (8, 16)]                  # NEW houses it has never seen

print("STEP 1. Data")
print(f"  train (model sees these): {train}")
print(f"  test  (hidden from model): {test}\n")


# STEP 2. A loss function (from the earlier lesson): average of (prediction - truth)^2
#         Small loss = good.  Big loss = bad.
def loss(model, data):
    total = 0
    for size, price in data:
        total += (model(size) - price) ** 2
    return total / len(data)


# STEP 3. MODEL 1 - the MEMORIZER. It just remembers the training answers.
#         For a new size, it returns the price of the closest size it remembers.
def memorizer(size):
    closest = train[0]
    for example in train:
        if abs(example[0] - size) < abs(closest[0] - size):
            closest = example
    return closest[1]


print("STEP 3. Memorizer (looks up the closest training house)")
print(f"  size 3 -> {memorizer(3)}   (training said 8, so it is perfect on train)")
print(f"  size 8 -> {memorizer(8)}   (closest seen size is 5, so it says 12; real is 16)\n")

# STEP 4. MODEL 2 - the RULE LEARNER: price = w * size.
#         It learns one number w (the pattern) instead of memorizing answers.
#         w = sum(size*price) / sum(size*size) is the lowest-loss w for this model
#         (the bottom of the loss bowl from the gradient lesson).
sum_sp = sum(size * price for size, price in train)
sum_ss = sum(size * size for size, price in train)
w = sum_sp / sum_ss


def rule(size):
    return w * size


print("STEP 4. Rule learner")
print(f"  learned w = {w:.2f}   (so price = {w:.2f} * size, close to the true 2)\n")


# STEP 5. MODEL 0 - the LAZY one: always predicts the average price. Ignores size.
average = sum(price for size, price in train) / len(train)


def lazy(size):
    return average


# STEP 6. COMPARE all three on TRAIN data and on TEST data.
print("STEP 6. Loss on train vs test (lower is better)")
print(f"  {'model':<12}{'train loss':>12}{'test loss':>12}")
for name, model in [("lazy", lazy), ("memorizer", memorizer), ("rule", rule)]:
    print(f"  {name:<12}{loss(model, train):>12.2f}{loss(model, test):>12.2f}")

# STEP 7. READ THE TABLE
print("""
STEP 7. What the table tells us
  lazy       bad on train AND bad on test   -> UNDERFITTING (too simple, learned nothing)
  memorizer  train loss is 0, test is worse -> OVERFITTING  (memorized, did not learn the pattern)
  rule       decent on both                 -> GOOD FIT     (learned the real pattern)

  The memorizer looks like the winner if you only check train data.
  Only the test data shows it is not really good.
""")

# KEY IDEAS
#   - Overfitting: low train loss, high test loss (big gap between them).
#   - Underfitting: high loss on both.
#   - Always judge a model on data it has never seen (the train/test split).
#   - Fixes: more data, a simpler model, early stopping, regularization.



# Train low and test low means a good model.
# Train low and test high means overfitting.
# Train high and test high means underfitting.
