// Two vectors
const v1 = tf.tensor([1, 2, 3]);
const v2 = tf.tensor([4, 5, 6]);

// Element-wise addition
const addResult = v1.add(v2);
console.log("Vector Addition:");
addResult.print();

// Element-wise multiplication
const mulResult = v1.mul(v2);
console.log("Vector Multiplication:");
mulResult.print();

// Reshape example
const t = tf.tensor([1, 2, 3, 4]);
const reshaped = t.reshape([2, 2]);
console.log("Reshaped Tensor:");
reshaped.print();

// Flatten example
const flattened = reshaped.flatten();
console.log("Flattened Tensor:");
flattened.print();
