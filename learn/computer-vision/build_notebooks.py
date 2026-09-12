"""Fill the Computer Vision exercise notebooks.

Answers from learntools/computer_vision/ex{1..6}. Notebook order matches module
order here, unlike Feature Engineering -- checked by reading each notebook's
import rather than assumed.

Most blanks are commented-out placeholders inside a keras.Sequential list
(`# ____,`) rather than a bare `____`, so the replacement has to keep the
surrounding model definition intact. These cells are written out whole.

These notebooks train real convnets, so the runs are slow and queue against
Kaggle's five-session limit.

    python kaggle/computer-vision/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

CELLS = {
"exercise-the-convolutional-classifier.ipynb": {
 7: '''# YOUR_CODE_HERE
# Freeze the base: its learned features are what we are borrowing, and a
# randomly initialised head would wreck them in the first few batches.
pretrained_base.trainable = False

# Check your answer
q_1.check()
''',
 10: '''from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    # YOUR CODE HERE. Attach a head of dense layers.
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Check your answer
q_2.check()
''',
 13: '''# YOUR CODE HERE: what loss function should you use for a binary
# classification problem? (Your answer for each should be a string.)
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Check your answer
q_3.check()
''',
},
"exercise-convolution-and-relu.ipynb": {
 8: '''# YOUR CODE HERE: Define a kernel with 3 rows and 3 columns.
# An edge detector: negative on one diagonal, positive on the other, so it
# responds to a change running across the image.
kernel = tf.constant([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
])
# Uncomment to view kernel
visiontools.show_kernel(kernel)

# Check your answer
q_1.check()
''',
 13: '''# YOUR CODE HERE: Give the TensorFlow convolution function (without arguments)
conv_fn = tf.nn.conv2d

# Check your answer
q_2.check()
''',
 18: '''# YOUR CODE HERE: Give the TensorFlow ReLU function (without arguments)
relu_fn = tf.nn.relu

# Check your answer
q_3.check()
''',
},
"exercise-maximum-pooling.ipynb": {
 6: '''# YOUR CODE HERE
# A 2x2 max window stepping 2 at a time: halves each spatial dimension and
# keeps the strongest activation in each patch.
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

# Check your answer
q_1.check()
''',
},
"exercise-custom-convnets.ipynb": {
 4: '''from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    # YOUR CODE HERE
    # Filters double again, and two convolutions before pooling let this block
    # build a more complex feature than a single one could.
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])

# Check your answer
q_1.check()
''',
},
"exercise-data-augmentation.ipynb": {
 17: '''from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),
    
    # Data Augmentation
    # Transformations that leave the label alone: a flipped or slightly turned
    # car is still a car. No vertical flip, which would not occur in this data.
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Check your answer
q_3.check()
''',
},
}

FILES = {
 "ex1": "exercise-the-convolutional-classifier.ipynb",
 "ex2": "exercise-convolution-and-relu.ipynb",
 "ex3": "exercise-maximum-pooling.ipynb",
 "ex4": "exercise-the-sliding-window.ipynb",
 "ex5": "exercise-custom-convnets.ipynb",
 "ex6": "exercise-data-augmentation.ipynb",
}


def main():
    for name, filename in FILES.items():
        nb = json.loads((HERE / "sources" / filename).read_text())
        edits = CELLS.get(filename, {})
        for index, new in sorted(edits.items()):
            cell = nb["cells"][index]
            text = "".join(cell["source"])
            if "____" not in text:
                raise SystemExit(f"{filename}: cell {index} has no ____ to fill")
            cell["source"] = new.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        out = HERE / name / filename
        out.write_text(json.dumps(nb, indent=1) + "\n")
        left = sum(1 for c in nb["cells"]
                   if c["cell_type"] == "code" and "____" in "".join(c["source"]))
        print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(edits)} cells, {left} ____ left)")


if __name__ == "__main__":
    main()
