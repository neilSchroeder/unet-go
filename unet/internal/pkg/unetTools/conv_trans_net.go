package unetTools

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// ConvParams represents the parameters for a transverse convolutional layer
type ConvTransParams struct {
	Activation    string
	InputChannels int
	KernelSize    int
	Stride        int
	NumFilters    int
	// and now for the AdamW optimizer
	beta1   float64
	beta2   float64
	epsilon float64
}

// ConvTransLayer represents a transverse convolutional layer
type ConvTransLayer struct {
	Weights       []*mat64.Dense
	Biases        []*mat64.Dense
	KernelSize    int
	Stride        int
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

// NewConvTransLayer initializes a new instance of ConvTransLayer
func NewConvTransLayer(InputChannels, KernelSize, Stride, NumFilters int, Activation string) *ConvTransLayer {
	// the number of filters determines the number of matrices used for the weights
	Weights := make([]*mat64.Dense, NumFilters)
	Biases := make([]*mat64.Dense, NumFilters)
	for i := 0; i < NumFilters; i++ {
		Weights[i] = mat64.NewDense(KernelSize, KernelSize, randomMatrixValues(KernelSize*KernelSize))
		Biases[i] = mat64.NewDense(1, 1, randomMatrixValues(1))
	}
	return &ConvTransLayer{
		Weights:       Weights,
		Biases:        Biases,
		KernelSize:    KernelSize,
		Stride:        Stride,
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

func (ctl *ConvTransLayer) Convolve(input, weights, biases *mat64.Dense) *mat64.Dense {
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

// Forward performs a forward pass through the ConvTransLayer
func (ctl *ConvTransLayer) Forward(input []*mat64.Dense) []*mat64.Dense {
	copy(ctl._input, input)
	layer_out := make([]*mat64.Dense, ctl.NumFilters)

	for i := 0; i < ctl.NumFilters; i++ {
		for j := 0; j < len(input); j++ {
			if j == 0 {
				layer_out[i] = ctl.Convolve(input[i], ctl.Weights[i], ctl.Biases[i])
			} else {
				layer_out[i].Add(layer_out[i], ctl.Convolve(input[i], ctl.Weights[i], ctl.Biases[i]))
			}
		}
		layer_out[i].Scale(1.0/float64(len(input)), layer_out[i])
		applyActivation(layer_out[i], ctl.Activation)
	}

	// Apply Activation function
	copy(ctl._output, layer_out)
	return layer_out
}

func (ctl *ConvTransLayer) TransverseConvolve(input, kernel, bias *mat64.Dense) *mat64.Dense {
	// output size
	in_rows, in_cols := input.Dims()
	out_rows := (in_rows-1)*ctl.Stride + ctl.KernelSize
	out_cols := (in_cols-1)*ctl.Stride + ctl.KernelSize
	// create output
	output := mat64.NewDense(out_rows, out_cols, nil)

	// fill output
	for i := 0; i < in_rows; i++ {
		out_i := i * ctl.Stride
		for j := 0; j < in_cols; j++ {
			out_j := j * ctl.Stride
			// copy input value to output
			for k := 0; k < ctl.KernelSize; k++ {
				for l := 0; l < ctl.KernelSize; l++ {
					value := output.At(out_i+k, out_j+l)
					value += kernel.At(k, l) * input.At(i, j)
					output.Set(out_i+k, out_j+l, value+bias.At(0, 0))
				}
			}
		}
	}
	return output
}

// Backward computes the backward pass of the convolutional layer.
// It takes the gradient of the output matrix as input,
// and returns the gradient of the input matrix, the gradient of the Weights matrix,
// and the gradient of the biases matrix.
func (ctl *ConvTransLayer) Backward(gradOutput *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	// TODO fix this
	return gradOutput, gradOutput
}

// UpdateWeightsAndBiases updates the Weights and biases of the convolutional layer using AdamW optimizer
// This function gets called after all the gradients have been computed and accumctlated.
func (ctl *ConvTransLayer) UpdateWeightsAndBiases(learningRate float64, gradWeights, gradBiases *mat64.Dense) {
	// Increment time step
	ctl.t++

	// Compute biased first and second moment estimates
	ctl.mWeights.Scale(ctl.beta1, ctl.mWeights)
	gradWeights.Scale(1-ctl.beta1, gradWeights)
	ctl.mWeights.Add(ctl.mWeights, gradWeights)
	ctl.vWeights.Scale(ctl.beta2, ctl.vWeights)
	gradWeightsPow2 := mat64.NewDense(gradWeights.RawMatrix().Rows, gradWeights.RawMatrix().Cols, nil)
	gradWeightsPow2.MulElem(gradWeights, gradWeights)
	gradWeightsPow2.Scale(1-ctl.beta2, gradWeightsPow2)
	ctl.vWeights.Add(ctl.vWeights, gradWeightsPow2)

	ctl.mBiases.Scale(ctl.beta1, ctl.mBiases)
	gradBiases.Scale(1-ctl.beta1, gradBiases)
	ctl.mBiases.Add(ctl.mBiases, gradBiases)
	ctl.vBiases.Scale(ctl.beta2, ctl.vBiases)
	gradBiasesPow2 := mat64.NewDense(gradBiases.RawMatrix().Rows, gradBiases.RawMatrix().Cols, nil)
	gradBiasesPow2.MulElem(gradBiases, gradBiases)
	gradBiasesPow2.Scale(1-ctl.beta2, gradBiasesPow2)
	ctl.vBiases.Add(ctl.vBiases, gradBiasesPow2)

	// Correct bias in the first moment
	mWeightsCorrected := mat64.NewDense(ctl.mWeights.RawMatrix().Rows, ctl.mWeights.RawMatrix().Cols, nil)
	mWeightsCorrected.Scale(1.0/(1-math.Pow(ctl.beta1, float64(ctl.t))), ctl.mWeights)

	mBiasesCorrected := mat64.NewDense(ctl.mBiases.RawMatrix().Rows, ctl.mBiases.RawMatrix().Cols, nil)
	mBiasesCorrected.Scale(1.0/(1-math.Pow(ctl.beta1, float64(ctl.t))), ctl.mBiases)

	// Correct bias in the second moment
	vWeightsCorrected := mat64.NewDense(ctl.vWeights.RawMatrix().Rows, ctl.vWeights.RawMatrix().Cols, nil)
	vWeightsCorrected.Scale(1.0/(1-math.Pow(ctl.beta2, float64(ctl.t))), ctl.vWeights)

	vBiasesCorrected := mat64.NewDense(ctl.vBiases.RawMatrix().Rows, ctl.vBiases.RawMatrix().Cols, nil)
	vBiasesCorrected.Scale(1.0/(1-math.Pow(ctl.beta2, float64(ctl.t))), ctl.vBiases)

	// Update Weights
	for _, weight_mat := range ctl.Weights {
		mWeightsCorrected.MulElem(mWeightsCorrected, ConstDivMatrix(learningRate, MatrixAddConst(MatrixSqrt(vWeightsCorrected), ctl.epsilon)))
		weight_mat.Sub(weight_mat, mWeightsCorrected)
	}

	// Update biases
	for _, bias_mat := range ctl.Biases {
		mBiasesCorrected.MulElem(mBiasesCorrected, ConstDivMatrix(learningRate, MatrixAddConst(MatrixSqrt(vBiasesCorrected), ctl.epsilon)))
		bias_mat.Sub(bias_mat, mBiasesCorrected)
	}
}

// Summary returns a summary of the ConvTransLayer
func (ctl *ConvTransLayer) Summary() string {
	summary := fmt.Sprintf("    Activation: %s\n", ctl.Activation)
	summary += fmt.Sprintf("    KernelSize: %d\n", ctl.KernelSize)
	summary += fmt.Sprintf("    InputChannels: %d\n", ctl.InputChannels)
	summary += fmt.Sprintf("    NumFilters: %d\n", ctl.NumFilters)
	return summary
}
