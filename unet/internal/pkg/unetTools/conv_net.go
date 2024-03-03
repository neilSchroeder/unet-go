package unetTools

import (
	"math"

	"github.com/gonum/matrix/mat64"
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
	Weights       *mat64.Dense
	Biases        *mat64.Dense
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
}

// NewConvLayer initializes a new instance of ConvLayer
func NewConvLayer(InputChannels, KernelSize, NumFilters int, Activation string) *ConvLayer {
	Weights := mat64.NewDense(KernelSize*KernelSize*InputChannels, NumFilters, randomMatrixValues(KernelSize*KernelSize*InputChannels*NumFilters))
	biases := mat64.NewDense(1, NumFilters, randomMatrixValues(NumFilters))
	return &ConvLayer{
		Weights:       Weights,
		Biases:        biases,
		KernelSize:    KernelSize,
		Activation:    Activation,
		InputChannels: InputChannels,
		NumFilters:    NumFilters,
		// and now for the AdamW optimizer
		beta1:    0.9,
		beta2:    0.999,
		epsilon:  1e-8,
		mWeights: mat64.NewDense(Weights.RawMatrix().Rows, Weights.RawMatrix().Cols, nil),
		vWeights: mat64.NewDense(Weights.RawMatrix().Rows, Weights.RawMatrix().Cols, nil),
		mBiases:  mat64.NewDense(biases.RawMatrix().Rows, biases.RawMatrix().Cols, nil),
		vBiases:  mat64.NewDense(biases.RawMatrix().Rows, biases.RawMatrix().Cols, nil),
		t:        0,
	}
}

// Forward performs a forward pass through the ConvLayer
func (cl *ConvLayer) Forward(input *mat64.Dense) *mat64.Dense {
	inputRows, inputCols := input.Dims()
	NumFilters, _ := cl.Weights.Dims()
	outputRows, outputCols := inputRows-cl.KernelSize+1, inputCols-cl.KernelSize+1
	output := mat64.NewDense(outputRows, outputCols, nil)

	// Perform convolution
	for i := 0; i < outputRows; i++ {
		for j := 0; j < outputCols; j++ {
			for k := 0; k < NumFilters; k++ {
				var sum float64
				for m := 0; m < cl.KernelSize; m++ {
					for n := 0; n < cl.KernelSize; n++ {
						for c := 0; c < cl.InputChannels; c++ {
							inputVal := input.At(i+m, j+n)
							weight := cl.Weights.At(m*cl.KernelSize*cl.InputChannels+n*cl.InputChannels+c, k)
							sum += inputVal * weight
						}
					}
				}
				bias := cl.Biases.At(0, k)
				sum += bias
				output.Set(i, j, sum)
			}
		}
	}

	// Apply Activation function
	applyActivation(output, cl.Activation)
	return output
}

// Backward computes the backward pass of the convolutional layer.
// It takes the gradient of the output matrix as input,
// and returns the gradient of the input matrix, the gradient of the Weights matrix,
// and the gradient of the biases matrix.
func (cl *ConvLayer) Backward(gradOutput *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	inputRows, inputCols := gradOutput.Dims()
	NumFilters, _ := cl.Weights.Dims()
	gradWeights := mat64.NewDense(cl.KernelSize*cl.KernelSize*cl.InputChannels, NumFilters, nil)
	gradBiases := mat64.NewDense(1, NumFilters, nil)

	// Compute the gradient of the loss with respect to the input
	for i := 0; i < inputRows; i++ {
		for j := 0; j < inputCols; j++ {
			for k := 0; k < NumFilters; k++ {
				gradBias := gradOutput.At(i, j)
				gradBiases.Set(0, k, gradBiases.At(0, k)+gradBias)
				for m := 0; m < cl.KernelSize; m++ {
					for n := 0; n < cl.KernelSize; n++ {
						for c := 0; c < cl.InputChannels; c++ {
							gradWeight := cl.Weights.At(m*cl.KernelSize*cl.InputChannels+n*cl.InputChannels+c, k) * gradOutput.At(i, j)
							gradWeights.Set(m*cl.KernelSize*cl.InputChannels+n*cl.InputChannels+c, k, gradWeights.At(m*cl.KernelSize*cl.InputChannels+n*cl.InputChannels+c, k)+gradWeight)
						}
					}
				}
			}
		}
	}

	return gradWeights, gradBiases
}

// UpdateWeightsAndBiases updates the Weights and biases of the convolutional layer using AdamW optimizer
// This function gets called after all the gradients have been computed and accumulated.
func (cl *ConvLayer) UpdateWeightsAndBiases(learningRate float64, gradWeights, gradBiases *mat64.Dense) {
	// Increment time step
	cl.t++

	// Compute biased first and second moment estimates
	cl.mWeights.Scale(cl.beta1, cl.mWeights)
	gradWeights.Scale(1-cl.beta1, gradWeights)
	cl.mWeights.Add(cl.mWeights, gradWeights)
	cl.vWeights.Scale(cl.beta2, cl.vWeights)
	gradWeightsPow2 := mat64.NewDense(gradWeights.RawMatrix().Rows, gradWeights.RawMatrix().Cols, nil)
	gradWeightsPow2.MulElem(gradWeights, gradWeights)
	gradWeightsPow2.Scale(1-cl.beta2, gradWeightsPow2)
	cl.vWeights.Add(cl.vWeights, gradWeightsPow2)

	cl.mBiases.Scale(cl.beta1, cl.mBiases)
	gradBiases.Scale(1-cl.beta1, gradBiases)
	cl.mBiases.Add(cl.mBiases, gradBiases)
	cl.vBiases.Scale(cl.beta2, cl.vBiases)
	gradBiasesPow2 := mat64.NewDense(gradBiases.RawMatrix().Rows, gradBiases.RawMatrix().Cols, nil)
	gradBiasesPow2.MulElem(gradBiases, gradBiases)
	gradBiasesPow2.Scale(1-cl.beta2, gradBiasesPow2)
	cl.vBiases.Add(cl.vBiases, gradBiasesPow2)

	// Correct bias in the first moment
	mWeightsCorrected := mat64.NewDense(cl.mWeights.RawMatrix().Rows, cl.mWeights.RawMatrix().Cols, nil)
	mWeightsCorrected.Scale(1.0/(1-math.Pow(cl.beta1, float64(cl.t))), cl.mWeights)

	mBiasesCorrected := mat64.NewDense(cl.mBiases.RawMatrix().Rows, cl.mBiases.RawMatrix().Cols, nil)
	mBiasesCorrected.Scale(1.0/(1-math.Pow(cl.beta1, float64(cl.t))), cl.mBiases)

	// Correct bias in the second moment
	vWeightsCorrected := mat64.NewDense(cl.vWeights.RawMatrix().Rows, cl.vWeights.RawMatrix().Cols, nil)
	vWeightsCorrected.Scale(1.0/(1-math.Pow(cl.beta2, float64(cl.t))), cl.vWeights)

	vBiasesCorrected := mat64.NewDense(cl.vBiases.RawMatrix().Rows, cl.vBiases.RawMatrix().Cols, nil)
	vBiasesCorrected.Scale(1.0/(1-math.Pow(cl.beta2, float64(cl.t))), cl.vBiases)

	// Update Weights
	Weights := mat64.DenseCopyOf(cl.Weights)
	mWeightsCorrected = ScaleMatrixByMatrix(mWeightsCorrected, ConstDivMatrix(learningRate, (MatrixAddConst(MatrixSqrt(vWeightsCorrected), cl.epsilon))))
	Weights.Sub(Weights, mWeightsCorrected)
	cl.Weights = Weights

	// Update biases
	biases := mat64.DenseCopyOf(cl.Biases)
	mBiasesCorrected = ScaleMatrixByMatrix(mBiasesCorrected, ConstDivMatrix(learningRate, (MatrixAddConst(MatrixSqrt(vBiasesCorrected), cl.epsilon))))
	biases.Sub(biases, mBiasesCorrected)
	cl.Biases = biases
}
