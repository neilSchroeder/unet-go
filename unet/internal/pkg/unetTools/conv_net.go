package unetTools

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
	mat "github.com/gonum/matrix/mat64"
)

// ConvParams represents the parameters for a convolutional layer
type ConvParams struct {
	Activation    string
	InputChannels int
	KernelSize    int
	NumFilters    int
	// and now for the AdamW optimizer
	beta1   float64
	beta2   float64
	epsilon float64
}

// ConvLayer represents a convolutional layer
type ConvLayer struct {
	Weights       []*mat64.Dense
	Biases        []*mat64.Dense
	KernelSize    int
	Activation    string
	InputChannels int
	NumFilters    int
	// and now for the AdamW optimizer
	beta1    float64
	beta2    float64
	epsilon  float64
	mWeights *mat64.Dense
	vWeights *mat64.Dense
	mBiases  *mat64.Dense
	vBiases  *mat64.Dense
	t        int
	_input   []*mat64.Dense
	_output  []*mat64.Dense
}

// NewConvLayer initializes a new instance of ConvLayer
func NewConvLayer(InputChannels, KernelSize, NumFilters int, Activation string) *ConvLayer {
	// the number of filters determines the number of matrices used for the weights
	Weights := make([]*mat64.Dense, NumFilters)
	Biases := make([]*mat64.Dense, NumFilters)
	for i := 0; i < NumFilters; i++ {
		Weights[i] = mat64.NewDense(KernelSize, KernelSize, randomMatrixValues(KernelSize*KernelSize))
		Biases[i] = mat64.NewDense(1, 1, randomMatrixValues(1))
	}
	return &ConvLayer{
		Weights:       Weights,
		Biases:        Biases,
		KernelSize:    KernelSize,
		Activation:    Activation,
		InputChannels: InputChannels,
		NumFilters:    NumFilters,
		// and now for the AdamW optimizer
		beta1:    0.9,
		beta2:    0.999,
		epsilon:  1e-8,
		mWeights: mat64.NewDense(KernelSize, KernelSize, nil),
		vWeights: mat64.NewDense(KernelSize, KernelSize, nil),
		mBiases:  mat64.NewDense(1, 1, nil),
		vBiases:  mat64.NewDense(1, 1, nil),
		t:        0,
	}

}

func (cl *ConvLayer) Convolve(input, weights, biases *mat64.Dense) *mat64.Dense {
	inputRows, inputCols := input.Dims()
	weightsRows, weightsCols := weights.Dims()
	outputRows := inputRows - weightsRows + 1
	outputCols := inputCols - weightsCols + 1
	output := mat64.NewDense(outputRows, outputCols, nil)

	for i := 0; i < outputRows; i++ {
		for j := 0; j < outputCols; j++ {
			sum := 0.0
			for m := 0; m < weightsRows; m++ {
				for n := 0; n < weightsCols; n++ {
					sum += input.At(i+m, j+n) * weights.At(m, n)
				}
			}
			sum /= float64(weightsRows * weightsCols)
			output.Set(i, j, sum+biases.At(0, 0))
		}
	}

	return output
}

// Forward performs a forward pass through the ConvLayer
func (cl *ConvLayer) Forward(input []*mat64.Dense) []*mat64.Dense {
	cl._input = input
	layer_out := make([]*mat64.Dense, cl.NumFilters)

	for i := 0; i < cl.NumFilters; i++ {
		for j := 0; j < len(input); j++ {
			if j == 0 {
				layer_out[i] = cl.Convolve(input[j], cl.Weights[i], cl.Biases[i])
			} else {
				layer_out[i].Add(layer_out[i], cl.Convolve(input[j], cl.Weights[i], cl.Biases[i]))
			}
		}
		layer_out[i].Scale(1.0/float64(len(input)), layer_out[i])
		applyActivation(layer_out[i], cl.Activation)
	}

	cl._output = layer_out
	return layer_out
}

// Backward computes the backward pass of the convolutional layer.
// It takes the gradient of the output matrix as input,
// and returns the gradient of the input matrix, the gradient of the Weights matrix,
// and the gradient of the biases matrix.
func (cl *ConvLayer) Backward(outputGrad *mat.Dense, learningRate float64) {
	// Iterate over each location in the output gradient
	fmt.Println("In ConvLayer.Backward:")
	fmt.Printf("OutputGrad size: %d x %d\n", outputGrad.RawMatrix().Rows, outputGrad.RawMatrix().Cols)
	fmt.Printf("Input size:      %d x %d\n", cl._input[0].RawMatrix().Rows, cl._input[0].RawMatrix().Cols)
	for i := 0; i < cl.NumFilters; i++ {
		// Initialize gradients of weights and biases

		gradWeights := make([]*mat64.Dense, len(cl._input))
		gradBiases := mat64.NewDense(1, 1, nil)
		for j := 0; j < len(cl._input); j++ {
			gradWeights[j] = mat64.NewDense(cl.KernelSize, cl.KernelSize, nil)
		}

		// Iterate over each location in the output gradient
		for outX := 0; outX < outputGrad.RawMatrix().Rows; outX++ {
			for outY := 0; outY < outputGrad.RawMatrix().Cols; outY++ {
				// Compute the gradients for each weight in the kefrnel
				for x := 0; x < cl.KernelSize; x++ {
					for y := 0; y < cl.KernelSize; y++ {
						for c := 0; c < len(cl._input); c++ {
							// Compute the gradient of the loss with respect to this weight
							gradWeights[c].Set(x, y, gradWeights[c].At(x, y)+
								outputGrad.At(outX, outY)*cl._input[c].At(outX+x, outY+y))
						}
					}
				}
				// Accumulate gradients for biases
				gradBiases.Set(0, 0, gradBiases.At(0, 0)+outputGrad.At(outX, outY))
			}
		}

		// Update weights and biases using AdamW optimizer
		fmt.Println("weights before:", cl.Weights[i])
		for c := 0; c < len(cl._input); c++ {
			cl.UpdateWeightsAndBiases(i, learningRate, gradWeights[c], gradBiases)
		}
		fmt.Println("weights after:", cl.Weights[i])
	}
	// resize gradOutput to have the same size as the input
	*outputGrad = *ResizeMatrix(outputGrad, cl._input[0].RawMatrix().Rows, cl._input[0].RawMatrix().Cols)
}

// UpdateWeightsAndBiases updates the Weights and biases of the convolutional layer using AdamW optimizer
// This function gets called after all the gradients have been computed and accumulated.
func (cl *ConvLayer) UpdateWeightsAndBiases(filterIndex int, learningRate float64, gradWeights, gradBiases *mat64.Dense) {
	if learningRate == 0 {
		// throw an error
		errorString := "Learning rate cannot be zero"
		panic(errorString)
	}
	// Compute AdamW updates for weights
	cl.mWeights.Apply(func(i, j int, v float64) float64 {
		return cl.beta1*v + (1-cl.beta1)*gradWeights.At(i, j)
	}, cl.mWeights)
	cl.vWeights.Apply(func(i, j int, v float64) float64 {
		return cl.beta2*v + (1-cl.beta2)*gradWeights.At(i, j)*gradWeights.At(i, j)
	}, cl.vWeights)
	mHat := mat.NewDense(gradWeights.RawMatrix().Rows, gradWeights.RawMatrix().Cols, nil)
	mHat.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(cl.beta1, float64(cl.t+1)))
	}, cl.mWeights)
	vHat := mat.NewDense(gradWeights.RawMatrix().Rows, gradWeights.RawMatrix().Cols, nil)
	vHat.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(cl.beta2, float64(cl.t+1)))
	}, cl.vWeights)
	weightUpdate := mat.NewDense(gradWeights.RawMatrix().Rows, gradWeights.RawMatrix().Cols, nil)
	for i := 0; i < gradWeights.RawMatrix().Rows; i++ {
		for j := 0; j < gradWeights.RawMatrix().Cols; j++ {
			weightUpdate.Set(i, j, learningRate*mHat.At(i, j)/(math.Sqrt(vHat.At(i, j))+cl.epsilon))
		}
	}
	// Update weights
	cl.Weights[filterIndex].Sub(cl.Weights[filterIndex], weightUpdate)

	// Compute AdamW updates for biases
	cl.mBiases.Apply(func(i, j int, v float64) float64 {
		return cl.beta1*v + (1-cl.beta1)*gradBiases.At(i, j)
	}, cl.mBiases)
	cl.vBiases.Apply(func(i, j int, v float64) float64 {
		return cl.beta2*v + (1-cl.beta2)*gradBiases.At(i, j)*gradBiases.At(i, j)
	}, cl.vBiases)
	mHatB := mat.NewDense(gradBiases.RawMatrix().Rows, gradBiases.RawMatrix().Cols, nil)
	mHatB.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(cl.beta1, float64(cl.t+1)))
	}, cl.mBiases)
	vHatB := mat.NewDense(gradBiases.RawMatrix().Rows, gradBiases.RawMatrix().Cols, nil)
	vHatB.Apply(func(i, j int, v float64) float64 {
		return v / (1 - math.Pow(cl.beta2, float64(cl.t+1)))
	}, cl.vBiases)
	biasUpdate := mat.NewDense(gradBiases.RawMatrix().Rows, gradBiases.RawMatrix().Cols, nil)
	biasUpdate.Apply(func(i, j int, v float64) float64 {
		return learningRate * v / (math.Sqrt(vHatB.At(i, j)) + cl.epsilon)
	}, mHatB)

	// Update biases
	cl.Biases[filterIndex].Sub(cl.Biases[filterIndex], biasUpdate)

	// Increment time step
	cl.t++
}

// Summary returns a summary of the ConvLayer
func (cl *ConvLayer) Summary() string {
	summary := fmt.Sprintf("    Activation: %s\n", cl.Activation)
	summary += fmt.Sprintf("    KernelSize: %d\n", cl.KernelSize)
	summary += fmt.Sprintf("    InputChannels: %d\n", cl.InputChannels)
	summary += fmt.Sprintf("    NumFilters: %d\n", cl.NumFilters)
	return summary
}
