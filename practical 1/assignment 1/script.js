const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

async function run() {
  await model.fit(xs, ys, { epochs: 150 });

  const preds = model.predict(xs);
  console.log("Actual Values:");
  ys.print();

  console.log("Predicted Values:");
  preds.print();
}

run();
