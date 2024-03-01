package unetTools

import (
	"github.com/gonum/matrix/mat64"
)

// DecoderParams represents the parameters for a decoder
type DecoderParams struct {
	ConvParams     []ConvParams
	UpsampleParams []UpsampleParams
}

// Decoder represents a decoder for convolutional neural networks
type Decoder struct {
	convLayers     []*ConvLayer
	upsampleLayers []*UpsampleLayer
}

// NewDecoder initializes a new instance of Decoder
func NewDecoder(inputChannels int, convParams []ConvParams, upsampleParams []UpsampleParams) *Decoder {
	decoder := &Decoder{}

	// Create upsampling layers
	for _, params := range upsampleParams {
		decoder.upsampleLayers = append(decoder.upsampleLayers, NewUpsampleLayer(params.ScaleFactor))
	}

	// Create convolutional layers
	for _, params := range convParams {
		decoder.convLayers = append(decoder.convLayers, NewConvLayer(inputChannels, params.KernelSize, params.NumFilters, params.Activation))
		inputChannels = params.NumFilters
	}

	return decoder
}

// Forward performs a forward pass through the Decoder
func (dec *Decoder) Forward(input *mat64.Dense) *mat64.Dense {
	output := input
	for i, upsampleLayer := range dec.upsampleLayers {
		// Forward pass through upsampling layer
		output = upsampleLayer.Forward(output)

		// Forward pass through convolutional layer
		output = dec.convLayers[i].Forward(output)
	}
	return output
}
