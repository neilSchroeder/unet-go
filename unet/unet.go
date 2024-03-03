package main

import (
	"github.com/gonum/matrix/mat64"

	"unet/unet/internal/pkg/unetTools"
)

func main() {
	// builds a simple U-Net model

	my_net = unetTools.NewUnet(
		572, // input size
		1,   // input channels
		3,   // number of encoder-decoder pairs
		8,   // num filters in first layer

	// Forward pass
	input := mat64.NewDense(572, 572, nil)
	output := encoder.Forward(input)
	output = decoder.Forward(output)

}
