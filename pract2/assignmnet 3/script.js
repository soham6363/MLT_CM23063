const MNISTData = (await import("./mnist.js")).MNISTData;

function setStatus(msg) {
  document.getElementById('status').innerText = 'Status: ' + msg;
}

function log(msg) {
  const r = document.getElementById('results');
  r.innerHTML += `<div>${msg}</div>`;
}

window.startComparison = async function () {
  document.getElementById('results').innerHTML = '<b>Results:</b>';
  setStatus('Loading MNIST data...');

  const data = new MNISTData();
  await data.load();

  const { xs: xsCNN, labels } = data.getTrainData();
  const xsDense = xsCNN.reshape([xsCNN.shape[0], 784]);

  // ── DENSE MODEL ──
  setStatus('Training Dense Network...');
  const dense = tf.sequential();
  dense.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [784] }));
  dense.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  dense.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  const denseHist = await dense.fit(xsDense, labels, {
    epochs: 3, batchSize: 128,
    callbacks: { onEpochEnd: (e, l) => log(`Dense Epoch ${e+1}/3 — Acc: ${(l.acc*100).toFixed(2)}%`) }
  });
  const denseAcc = (denseHist.history.acc.pop() * 100).toFixed(2);

  // ── CNN MODEL ──
  setStatus('Training CNN...');
  const cnn = tf.sequential();
  cnn.add(tf.layers.conv2d({ inputShape: [28,28,1], filters: 16, kernelSize: 3, activation: 'relu' }));
  cnn.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  cnn.add(tf.layers.flatten());
  cnn.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  cnn.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  const cnnHist = await cnn.fit(xsCNN, labels, {
    epochs: 3, batchSize: 128,
    callbacks: { onEpochEnd: (e, l) => log(`CNN Epoch ${e+1}/3 — Acc: ${(l.acc*100).toFixed(2)}%`) }
  });
  const cnnAcc = (cnnHist.history.acc.pop() * 100).toFixed(2);

  // ── RESULT ──
  log(`<br><b>🏆 Final Results:</b>`);
  log(`Dense Network Accuracy: <b>${denseAcc}%</b>`);
  log(`CNN Accuracy: <b>${cnnAcc}%</b>`);
  log(`Winner: <b>${parseFloat(cnnAcc) > parseFloat(denseAcc) ? '🥇 CNN' : '🥇 Dense'}</b>`);
  setStatus('Done!');
};