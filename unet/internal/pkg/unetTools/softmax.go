package unetTools

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// SoftmaxLayer represents a softmax layer
type SoftmaxLayer struct {
	inputSize  int
	outputSize int
}

// NewSoftmaxLayer initializes a new instance of SoftmaxLayer
func NewSoftmaxLayer(inputSize, outputSize int) *SoftmaxLayer {
	softmaxLayer := &SoftmaxLayer{
		inputSize:  inputSize,
		outputSize: outputSize,
	}
	return softmaxLayer
}

// Forward performs a forward pass through the SoftmaxLayer
func (sl *SoftmaxLayer) Forward(input *mat64.Dense) *mat64.Dense {
	numRows, numCols := input.Dims()
	output := mat64.NewDense(numRows, numCols, nil)

	// Iterate over each row of the input matrix
	for i := 0; i < numRows; i++ {
		// Compute the maximum score in the row
		maxScore := input.At(i, 0)
		for j := 1; j < numCols; j++ {
			if input.At(i, j) > maxScore {
				maxScore = input.At(i, j)
			}
		}

		// Compute the sum of exponentials of scores (for numerical stability)
		sumExpScores := 0.0
		for j := 0; j < numCols; j++ {
			sumExpScores += math.Exp(input.At(i, j) - maxScore)
		}

		// Compute the softmax scores for the row
		for j := 0; j < numCols; j++ {
			output.Set(i, j, math.Exp(input.At(i, j)-maxScore)/sumExpScores)
		}
	}

	return output
}
