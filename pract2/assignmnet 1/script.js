import { MNISTData } from "./mnist.js";

function log(msg) {
  const logDiv = document.getElementById('log');
  logDiv.innerHTML += `<div>${msg}</div>`;
  logDiv.scrollTop = logDiv.scrollHeight;
}

function setStatus(msg) {
  document.getElementById('status').innerText = 'Status: ' + msg;
}

window.startTraining = async function () {
  setStatus('Loading MNIST data...');
  log('📦 Loading MNIST data...');

  const data = new MNISTData();
  await data.load();
  const { xs, labels } = data.getTrainData();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  setStatus('Training...');
  log('🚀 Training started...');

  await model.fit(xs, labels, {
    epochs: 5,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        log(`✅ Epoch ${epoch + 1}/5 — Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(2)}%`);
      }
    }
  });

  setStatus('Training Complete!');
  log('🎉 Training Complete!');
}