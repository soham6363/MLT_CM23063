// Scalar (0D Tensor)
const scalar = tf.scalar(7);
console.log("Scalar:");
scalar.print();

// Vector (1D Tensor)
const vector = tf.tensor([1, 2, 3]);
console.log("Vector:");
vector.print();

// Matrix (2D Tensor)
const matrix = tf.tensor([
  [1, 2],
  [3, 4]
]);
console.log("Matrix:");
matrix.print();
