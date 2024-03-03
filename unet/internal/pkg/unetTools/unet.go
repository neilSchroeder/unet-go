package unetTools

import (
	"slices"

	"github.com/gonum/matrix/mat64"
)

// UnetParams represents the parameters for a U-Net model
type UnetParams struct {
	EncoderParams
	DecoderParams
}

// Unet represents a U-Net model
type Unet struct {
	inputSize     int    // Size of input (assumed to be square)
	inputChannels int    // Number of input channels
	numEnDecoders int    // Number of encoder-decoder pairs
	maxNumFilters int    // Maximum number of filters in conv layers
	activation    string // Activation function
	kernelSize    int    // Size of convolutional kernel

	encoders   []*Encoder
	bottleneck *Decoder
	decoders   []*Decoder
}

// NewUnet initializes a new instance of Unet
func NewUnet(
	inputSize int,
	inputChannels int,
	numEnDecoders int,
	maxNumFilters int,
	activation string,
	kernelSize int,
) *Unet {

	cl_params := ConvParams{
		activation,
		inputChannels,
		kernelSize,
		maxNumFilters,
		0.9,
		0.999,
		1e-8,
	}

	unet := &Unet{
		inputSize:     inputSize,
		inputChannels: inputChannels,
		numEnDecoders: numEnDecoders,
		maxNumFilters: maxNumFilters,
		activation:    activation,
		kernelSize:    kernelSize,

		encoders: make([]*Encoder, numEnDecoders),
		bottleneck: NewDecoder(
			[]ConvParams{cl_params, cl_params},
			[]UpsampleParams{{kernelSize}},
		),
		decoders: make([]*Decoder, numEnDecoders),
	}

	// build the encoder-decoder pairs
	for i := 0; i < numEnDecoders; i++ {
		unet.encoders[i] = NewEncoder(
			[]ConvParams{cl_params, cl_params},
			[]PoolParams{{kernelSize, kernelSize}},
		)
		unet.decoders[i] = NewDecoder(
			[]ConvParams{cl_params, cl_params},
			[]UpsampleParams{{kernelSize}},
		)
		cl_params.NumFilters *= 2

	}
	unet.bottleneck = NewDecoder(
		[]ConvParams{cl_params, cl_params},
		[]UpsampleParams{{1}},
	)
	// reverse unet.decoders so that the densest layer is first
	slices.Reverse(unet.decoders)

	return unet
}

// Forward performs a forward pass through the U-Net model
func (unet *Unet) Forward(input *mat64.Dense) {

	// pass through encoders
	encoder_output := make([]*mat64.Dense, unet.numEnDecoders)
	for i := 0; i < unet.numEnDecoders; i++ {
		encoder_output[i] = unet.encoders[i].Forward(input)
		input = encoder_output[i]
	}

	// handle bottleneck
	input = unet.bottleneck.Forward(input, encoder_output[unet.numEnDecoders-1])

	// pass through decoders
	for i := 0; i < unet.numEnDecoders; i++ {
		input = unet.decoders[i].Forward(input, encoder_output[i])
	}
	// no return, pass by reference
}

// Backward performs a backward pass through the U-Net model
func (unet *Unet) Backward(input *mat64.Dense, target *mat64.Dense) *mat64.Dense {

	// compute loss
	loss := mat64.NewDense(0, 0, nil)
	loss.Sub(input, target)

	// compute gradients for both weights and biases
	gradOutput := loss
	for i := 0; i < unet.numEnDecoders; i++ {
		gradOutput = unet.decoders[i].Backward(gradOutput)
	}
	gradOutput, _, _ = unet.middle_layer.Backward(gradOutput)
	for i := 0; i < unet.numEnDecoders; i++ {
		gradOutput = unet.encoders[i].Backward(gradOutput)
	}

	return gradOutput
}
