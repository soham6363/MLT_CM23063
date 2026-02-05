// Generate synthetic data: y = 2x + 1
const xVals = tf.tensor1d([1, 2, 3, 4, 5]);
const yVals = tf.tensor1d([3, 5, 7, 9, 11]);

// Define model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compile model
model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

// Train model
async function trainModel() {
  await model.fit(xVals, yVals, {
    epochs: 200,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0)
          console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      }
    }
  });

  // Test prediction
  const prediction = model.predict(tf.tensor1d([6]));
  console.log("Prediction for x=6:");
  prediction.print();
}

trainModel();
