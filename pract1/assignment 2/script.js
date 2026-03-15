const xs = tf.tensor1d([1, 2, 3, 4, 5]);
const ys = tf.tensor1d([3, 5, 7, 9, 11]);

async function trainWithLR(lr) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(lr),
    loss: 'meanSquaredError'
  });

  console.log(`Training with Learning Rate: ${lr}`);
  await model.fit(xs, ys, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch} Loss: ${logs.loss}`);
      }
    }
  });
}

trainWithLR(0.01);
trainWithLR(0.1);
trainWithLR(0.5);
