package main

import (
	"fmt"
	"unet/unet/internal/pkg/unetTools"
)

func main() {
	// builds a simple U-Net model

	my_net := unetTools.NewUnet(
		572,                // input size
		1,                  // input channels
		4,                  // number of encoder-decoder pairs
		64,                 // maximum number of filters in conv layers
		"sigmoid",          // activation function
		3,                  // size of convolutional kernel
		2,                  // size of pooling kernel
		2,                  // stride of pooling kernel
		unetTools.DiceLoss, // loss function
	)
	my_net.Summary()

	// load image from data/ directory
	input := unetTools.LoadImage("/home/nschroed/work/unet-go/unet/data/CHANEL_THUMB.jpg")

	my_net.Step(input, input, 0.001)
	fmt.Print(my_net.GetLoss())

}
