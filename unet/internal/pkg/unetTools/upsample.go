package unetTools

import (
	"github.com/gonum/matrix/mat64"
)

// UpsampleParams represents the parameters for an upsampling layer
type UpsampleParams struct {
	ScaleFactor int
}

// UpsampleLayer represents an upsampling layer
type UpsampleLayer struct {
	scaleFactor int
}

// NewUpsampleLayer initializes a new instance of UpsampleLayer
func NewUpsampleLayer(scaleFactor int) *UpsampleLayer {
	return &UpsampleLayer{
		scaleFactor: scaleFactor,
	}
}

// Forward performs a forward pass through the UpsampleLayer
func (ul *UpsampleLayer) Forward(input *mat64.Dense) *mat64.Dense {
	inputRows, inputCols := input.Dims()
	outputRows := inputRows * ul.scaleFactor
	outputCols := inputCols * ul.scaleFactor
	output := mat64.NewDense(outputRows, outputCols, nil)

	for i := 0; i < inputRows; i++ {
		for j := 0; j < inputCols; j++ {
			val := input.At(i, j)
			for m := 0; m < ul.scaleFactor; m++ {
				for n := 0; n < ul.scaleFactor; n++ {
					output.Set(i*ul.scaleFactor+m, j*ul.scaleFactor+n, val)
				}
			}
		}
	}

	return output
}

// Backward performs a backward pass through the UpsampleLayer
func (ul *UpsampleLayer) Backward(gradOutput *mat64.Dense) *mat64.Dense {
	return gradOutput
}
