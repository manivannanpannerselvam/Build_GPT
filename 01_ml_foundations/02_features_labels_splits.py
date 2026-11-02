"""Level 1 - Concept 2: Features, labels, and train/validation/test split

Pure Python, no libraries. Run:  python3 02_features_labels_splits.py

Task: predict a house price from its size and number of bedrooms.
"""
import random

# 1. FEATURES (X) and LABEL (y)
#    Feature = an input the model looks at.   Label = the answer we want to predict.
#    Each row is one example: (size, bedrooms) -> price
houses = [
    # (size, bedrooms, price)
    (1, 1, 2), (2, 1, 4), (3, 2, 6), (4, 2, 8), (5, 3, 10),
    (6, 3, 12), (7, 3, 14), (8, 4, 16), (9, 4, 18), (10, 5, 20),
    (11, 5, 22), (12, 5, 24), (13, 6, 26), (14, 6, 28), (15, 6, 30),
    (16, 7, 32), (17, 7, 34), (18, 8, 36), (19, 8, 38), (20, 8, 40),
]

X = [(size, beds) for size, beds, price in houses]   # features
y = [price for size, beds, price in houses]          # labels

print("1. Features and labels")
print(f"  example 0: features X = {X[0]}  ->  label y = {y[0]}")
print(f"  total examples: {len(X)}\n")

# 2. WHY SPLIT?
#    If we test the model on the same data it trained on, it can just
#    memorize the answers and still look perfect. We need unseen data
#    to check if it really learned the rule.
#
#      TRAIN      (60%)  the model learns from this
#      VALIDATION (20%)  we check progress and tune settings with this
#      TEST       (20%)  final exam, used ONCE at the very end

# 3. SHUFFLE FIRST
#    Our data is sorted by size. Without shuffling, the test set would
#    hold only the biggest houses. A fixed seed makes results repeatable.
random.seed(42)
indices = list(range(len(X)))
random.shuffle(indices)

# 4. SPLIT by position in the shuffled list
n = len(indices)
train_end = int(n * 0.6)          # 60%
val_end = int(n * 0.8)            # 60% + 20%

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]


def pick(idx_list):
    return [X[i] for i in idx_list], [y[i] for i in idx_list]


X_train, y_train = pick(train_idx)
X_val, y_val = pick(val_idx)
X_test, y_test = pick(test_idx)

print("2. Split sizes")
print(f"  train: {len(X_train)}   validation: {len(X_val)}   test: {len(X_test)}")

# 5. CHECK: no example appears in two sets (that would be "data leakage")
assert not set(train_idx) & set(val_idx)
assert not set(train_idx) & set(test_idx)
assert not set(val_idx) & set(test_idx)
print("  no overlap between sets: OK\n")

# 6. USE THE SPLITS
#    Train: learn w (same nudging idea as Concept 1, on feature 'size').
#    Validation: check how well it does on unseen data.
#    Test: final score, once.


def predict(size, w):
    return w * size


def loss(X_part, y_part, w):
    total = 0
    for (size, beds), true_y in zip(X_part, y_part):
        total += (predict(size, w) - true_y) ** 2
    return total / len(X_part)


w = 1.5
step = 0.1
for _ in range(50):
    current = loss(X_train, y_train, w)          # learn from TRAIN only
    if loss(X_train, y_train, w + step) < current:
        w += step
    elif loss(X_train, y_train, w - step) < current:
        w -= step

print("3. Results")
print(f"  learned w = {w:.2f}")
print(f"  train loss:      {loss(X_train, y_train, w):.3f}")
print(f"  validation loss: {loss(X_val, y_val, w):.3f}   <- used to tune/choose models")
print(f"  test loss:       {loss(X_test, y_test, w):.3f}   <- final score, look once")

# KEY RULES
#   - Learn only from TRAIN.
#   - Use VALIDATION to compare settings (learning rate, model type...).
#   - Touch TEST once at the end; otherwise you are tuning to the exam.
#   - Train loss low but validation loss high = overfitting (Concept 4).
