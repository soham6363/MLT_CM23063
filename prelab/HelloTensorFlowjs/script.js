console.log("TensorFlow.js Version:", tf.version.tfjs);

const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([4, 5, 6]);

const sum = a.add(b);

console.log("Addition Result:");
sum.print();
