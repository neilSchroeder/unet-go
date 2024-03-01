package unetTools

import (
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// FullyConnectedLayer represents a fully connected layer
type FullyConnectedLayer struct {
	weights    *mat64.Dense
	biases     *mat64.Dense
	activation string
}

// NewFullyConnectedLayer initializes a new instance of FullyConnectedLayer
func NewFullyConnectedLayer(inputSize, outputSize int, activation string) *FullyConnectedLayer {
	weights := mat64.NewDense(inputSize, outputSize, randomMatrixValues(inputSize*outputSize))
	biases := mat64.NewDense(1, outputSize, randomMatrixValues(outputSize))
	return &FullyConnectedLayer{
		weights:    weights,
		biases:     biases,
		activation: activation,
	}
}

// Forward performs a forward pass through the FullyConnectedLayer
func (fcl *FullyConnectedLayer) Forward(input *mat64.Dense) *mat64.Dense {
	numRows, _ := input.Dims()
	outputSize, _ := fcl.weights.Dims()
	output := mat64.NewDense(numRows, outputSize, nil)

	// Perform matrix multiplication: input * weights
	output.Mul(input, fcl.weights)

	// Add biases
	output.Add(output, fcl.biases)

	// Apply activation function
	applyActivation(output, fcl.activation)

	return output
}

// applyActivation applies the specified activation function to each element of the matrix
func applyActivation(matrix *mat64.Dense, activation string) {
	numRows, numCols := matrix.Dims()
	switch activation {
	case "relu":
		for i := 0; i < numRows; i++ {
			for j := 0; j < numCols; j++ {
				if matrix.At(i, j) < 0 {
					matrix.Set(i, j, 0)
				}
			}
		}
	case "sigmoid":
		for i := 0; i < numRows; i++ {
			for j := 0; j < numCols; j++ {
				matrix.Set(i, j, 1/(1+math.Exp(-matrix.At(i, j))))
			}
		}
	}
}

// randomMatrixValues generates random values for a matrix of the specified size
func randomMatrixValues(size int) []float64 {
	values := make([]float64, size)
	for i := range values {
		values[i] = rand.Float64()
	}
	return values
}
