package unetTools

import (
	"github.com/gonum/matrix/mat64"
)

// MaxPoolParams represents the parameters for a max pooling layer
type PoolParams struct {
	PoolSize int
	Stride   int
}

// MaxPoolLayer represents a max pooling layer
type MaxPoolLayer struct {
	poolSize int
	stride   int
}

// NewMaxPoolLayer initializes a new instance of MaxPoolLayer
func NewMaxPoolLayer(poolSize, stride int) *MaxPoolLayer {
	return &MaxPoolLayer{
		poolSize: poolSize,
		stride:   stride,
	}
}

// Forward performs a forward pass through the MaxPoolLayer
func (mpl *MaxPoolLayer) Forward(input *mat64.Dense) *mat64.Dense {
	inputRows, inputCols := input.Dims()
	outputRows := (inputRows-mpl.poolSize)/mpl.stride + 1
	outputCols := (inputCols-mpl.poolSize)/mpl.stride + 1
	output := mat64.NewDense(outputRows, outputCols, nil)

	for i := 0; i < outputRows; i++ {
		for j := 0; j < outputCols; j++ {
			maxVal := -1e9 // initialize with a very small number
			for m := 0; m < mpl.poolSize; m++ {
				for n := 0; n < mpl.poolSize; n++ {
					val := input.At(i*mpl.stride+m, j*mpl.stride+n)
					if val > maxVal {
						maxVal = val
					}
				}
			}
			output.Set(i, j, maxVal)
		}
	}

	return output
}

// Backward performs a backward pass through the MaxPoolLayer
func (mpl *MaxPoolLayer) Backward(gradInput *mat64.Dense) *mat64.Dense {
	return gradInput
}
