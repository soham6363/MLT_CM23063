const xVals = tf.tensor1d([1, 2, 3, 4, 5]);
const yVals = tf.tensor1d([3, 5, 7, 9, 11]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

async function trainModel() {
  await model.fit(xVals, yVals, {
    epochs: 200,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0) {
          console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
          document.getElementById('loss').innerHTML =
            `⏳ Training... Epoch ${epoch}/200 | Loss: ${logs.loss.toFixed(6)}`;
        }
      }
    }
  });

  document.getElementById('loss').innerHTML = `✅ Training Complete! (200 epochs)`;

  const predTensor = model.predict(tf.tensor2d([[6]]));
  const predValue = predTensor.dataSync()[0];

  document.getElementById('output').innerHTML =
    `✅ Prediction for x=6: <strong>${predValue.toFixed(4)}</strong> (expected ≈ 13)`;

  console.log("Prediction for x=6:", predValue);
}

trainModel();