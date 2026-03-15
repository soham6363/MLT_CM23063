const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

async function predictUnseen() {
  await model.fit(xs, ys, { epochs: 200 });

  const unseen = tf.tensor2d([6, 7, 8], [3, 1]);
  const predictions = model.predict(unseen);

  console.log("Unseen Inputs:");
  unseen.print();

  console.log("Predicted Outputs:");
  predictions.print();

  console.log("Expected Outputs:");
  tf.tensor1d([13, 15, 17]).print();
}

predictUnseen();
