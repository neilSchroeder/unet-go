package unetTools

import (
	"github.com/gonum/matrix/mat64"
)

// ConvParams represents the parameters for a convolutional layer
type ConvParams struct {
	KernelSize int
	NumFilters int
	Activation string
}

// ConvLayer represents a convolutional layer
type ConvLayer struct {
	weights       *mat64.Dense
	biases        *mat64.Dense
	kernelSize    int
	activation    string
	inputChannels int
}

// NewConvLayer initializes a new instance of ConvLayer
func NewConvLayer(inputChannels, kernelSize, numFilters int, activation string) *ConvLayer {
	weights := mat64.NewDense(kernelSize*kernelSize*inputChannels, numFilters, randomMatrixValues(kernelSize*kernelSize*inputChannels*numFilters))
	biases := mat64.NewDense(1, numFilters, randomMatrixValues(numFilters))
	return &ConvLayer{
		weights:       weights,
		biases:        biases,
		kernelSize:    kernelSize,
		activation:    activation,
		inputChannels: inputChannels,
	}
}

// Forward performs a forward pass through the ConvLayer
func (cl *ConvLayer) Forward(input *mat64.Dense) *mat64.Dense {
	inputRows, inputCols := input.Dims()
	numFilters, _ := cl.weights.Dims()
	outputRows, outputCols := inputRows-cl.kernelSize+1, inputCols-cl.kernelSize+1
	output := mat64.NewDense(outputRows, outputCols, nil)

	// Perform convolution
	for i := 0; i < outputRows; i++ {
		for j := 0; j < outputCols; j++ {
			for k := 0; k < numFilters; k++ {
				var sum float64
				for m := 0; m < cl.kernelSize; m++ {
					for n := 0; n < cl.kernelSize; n++ {
						for c := 0; c < cl.inputChannels; c++ {
							inputVal := input.At(i+m, j+n)
							weight := cl.weights.At(m*cl.kernelSize*cl.inputChannels+n*cl.inputChannels+c, k)
							sum += inputVal * weight
						}
					}
				}
				bias := cl.biases.At(0, k)
				sum += bias
				output.Set(i, j, sum)
			}
		}
	}

	// Apply activation function
	applyActivation(output, cl.activation)
	return output
}
