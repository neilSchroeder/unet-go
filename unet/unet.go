package main

import (
	"github.com/gonum/matrix/mat64"

	"unet/unet/internal/pkg/unetTools"
)

func main() {
	// builds a simple U-Net model

	// Encoder
	convParams := []unetTools.ConvParams{
		{KernelSize: 3, NumFilters: 64, Activation: "relu"},
		{KernelSize: 3, NumFilters: 64, Activation: "relu"},
	}
	poolParams := []unetTools.PoolParams{
		{PoolSize: 2, Stride: 2},
		{PoolSize: 2, Stride: 2},
	}
	encoder := unetTools.NewEncoder(1, convParams, poolParams)

	// Decoder
	convParams = []unetTools.ConvParams{
		{KernelSize: 3, NumFilters: 64, Activation: "relu"},
		{KernelSize: 3, NumFilters: 64, Activation: "relu"},
	}
	upsampleParams := []unetTools.UpsampleParams{
		{ScaleFactor: 2},
		{ScaleFactor: 2},
	}
	decoder := unetTools.NewDecoder(64, convParams, upsampleParams)

	// Forward pass
	input := mat64.NewDense(572, 572, nil)
	output := encoder.Forward(input)
	output = decoder.Forward(output)

}
