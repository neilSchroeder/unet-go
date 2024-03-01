package unetTools

import (
	"github.com/gonum/matrix/mat64"
)

// EncoderParams represents the parameters for an encoder
type EncoderParams struct {
	ConvParams []ConvParams
	PoolParams []PoolParams
}

// Encoder represents an encoder for convolutional neural networks
type Encoder struct {
	convLayers []*ConvLayer
	poolLayers []*MaxPoolLayer
}

// NewEncoder initializes a new instance of Encoder
func NewEncoder(inputChannels int, convParams []ConvParams, poolParams []PoolParams) *Encoder {
	encoder := &Encoder{}

	// Create convolutional layers
	for _, params := range convParams {
		encoder.convLayers = append(encoder.convLayers, NewConvLayer(inputChannels, params.KernelSize, params.NumFilters, params.Activation))
		inputChannels = params.NumFilters
	}

	// Create pooling layers
	for _, params := range poolParams {
		encoder.poolLayers = append(encoder.poolLayers, NewMaxPoolLayer(params.PoolSize, params.Stride))
	}

	return encoder
}

// Forward performs a forward pass through the Encoder
func (enc *Encoder) Forward(input *mat64.Dense) *mat64.Dense {
	output := input
	for i, convLayer := range enc.convLayers {
		// Forward pass through convolutional layer
		output = convLayer.Forward(output)

		// If there is a pooling layer corresponding to this convolutional layer, apply pooling
		if i < len(enc.poolLayers) {
			output = enc.poolLayers[i].Forward(output)
		}
	}
	return output
}
