package unetTools

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// UpsampleParams represents the parameters for an upsampling layer
type UpsampleParams struct {
	kernelSize int
	stride     int
}

// UpsampleLayer represents an upsampling layer
type UpsampleLayer struct {
	kernelSize int
	stride     int
}

// NewUpsampleLayer initializes a new instance of UpsampleLayer
func NewUpsampleLayer(kernelSize int, stride int) *UpsampleLayer {
	return &UpsampleLayer{
		kernelSize: kernelSize,
		stride:     stride,
	}
}

// Forward performs a forward pass through the UpsampleLayer
// by applying a kernelSize x kernelSize upsampling kernel
// with stride stride
func (ul *UpsampleLayer) Forward(input *mat64.Dense) *mat64.Dense {
	// output size
	in_rows, in_cols := input.Dims()
	out_rows := (in_rows-1)*ul.stride + ul.kernelSize
	out_cols := (in_cols-1)*ul.stride + ul.kernelSize
	// create output
	output := mat64.NewDense(out_rows, out_cols, nil)
	kernel := getUpsampleKernel(ul.kernelSize)

	// fill output
	for i := 0; i < in_rows; i++ {
		out_i := i * ul.stride
		for j := 0; j < in_cols; j++ {
			out_j := j * ul.stride
			// copy input value to output
			for k := 0; k < ul.kernelSize; k++ {
				for l := 0; l < ul.kernelSize; l++ {
					value := output.At(out_i+k, out_j+l)
					value += kernel.At(k, l) * input.At(i, j)
					output.Set(out_i+k, out_j+l, value)
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

// getUpsampleKernel returns a kernel for upsampling
func getUpsampleKernel(kernelSize int) *mat64.Dense {
	// create kernel
	kernel := mat64.NewDense(kernelSize, kernelSize, nil)
	// TODO look for better kernels
	// fill kernel
	total := 0.0
	for i := 0; i < kernelSize; i++ {
		for j := 0; j < kernelSize; j++ {
			value := 1.0
			if (i+j)%2 == 1 {
				value = 2.0
			}
			total += value
			kernel.Set(i, j, value)
		}
	}
	kernel.Scale(1/total, kernel)
	return kernel
}

// Summary returns a summary of the UpsampleLayer
func (ul *UpsampleLayer) Summary() string {
	ret := "  UpsampleLayer:\n"
	ret += fmt.Sprintf("    kernelSize: %d\n", ul.kernelSize)
	ret += fmt.Sprintf("    stride: %d\n", ul.stride)
	return ret
}
