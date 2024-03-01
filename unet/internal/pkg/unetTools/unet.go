package unetTools

import (
	"math"
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

	encoders     []*Encoder
	middle_layer *ConvLayer
	decoders     []*Decoder
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

	unet := &Unet{
		inputSize:     inputSize,
		inputChannels: inputChannels,
		numEnDecoders: numEnDecoders,
		maxNumFilters: maxNumFilters,
		activation:    activation,
		kernelSize:    kernelSize,

		encoders:     make([]*Encoder, numEnDecoders),
		middle_layer: NewConvLayer(inputChannels, kernelSize, maxNumFilters, activation),
		decoders:     make([]*Decoder, numEnDecoders),
	}

	// build the encoder-decoder pairs
	for i := 0; i < numEnDecoders; i++ {
		num_filters := int(float64(maxNumFilters) / math.Pow(float64(kernelSize), float64(numEnDecoders-i-1)))
		unet.encoders[i] = NewEncoder(
			inputChannels,
			[]ConvParams{{kernelSize, num_filters, activation}},
			[]PoolParams{{kernelSize, kernelSize}},
		)
		unet.decoders[i] = NewDecoder(
			num_filters,
			[]ConvParams{{kernelSize, num_filters, activation}},
			[]UpsampleParams{{kernelSize}},
		)

	}
	// reverse unet.decoders
	slices.Reverse(unet.decoders)

	return unet
}

// Forward performs a forward pass through the U-Net model
func (unet *Unet) Forward(input *mat64.Dense) *mat64.Dense {

	/*
		An example with 3 conv layers would be
		conv1 -> pool1 -> conv2 -> pool2 -> conv3 ->
		upsample1 -> conv4 -> upsample2 -> conv5 -> output
	*/

	for i := 0; i < unet.numEnDecoders; i++ {
		input = unet.encoders[i].Forward(input)
	}
	input = unet.middle_layer.Forward(input)
	for i := 0; i < unet.numEnDecoders; i++ {
		input = unet.decoders[i].Forward(input)
	}
	return input

}
