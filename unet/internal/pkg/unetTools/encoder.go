package unetTools

import (
	"fmt"

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

	// internal params
	_dWeights []*mat64.Dense
	_dBiases  []*mat64.Dense
}

// NewEncoder initializes a new instance of Encoder
func NewEncoder(convParams []ConvParams, poolParams []PoolParams) *Encoder {
	encoder := &Encoder{}

	// Create convolutional layers
	encoder.convLayers = make([]*ConvLayer, len(convParams))
	for i, params := range convParams {
		encoder.convLayers[i] = NewConvLayer(
			params.InputChannels,
			params.KernelSize,
			params.NumFilters,
			params.Activation,
		)
	}

	// Create pooling layer
	encoder.poolLayers = make([]*MaxPoolLayer, len(poolParams))
	for i, params := range poolParams {
		encoder.poolLayers[i] = NewMaxPoolLayer(params.PoolSize, params.Stride)
	}

	encoder._dWeights = make([]*mat64.Dense, len(convParams))
	encoder._dBiases = make([]*mat64.Dense, len(convParams))

	return encoder
}

// Forward performs a forward pass through the Encoder
func (enc *Encoder) Forward(input []*mat64.Dense) []*mat64.Dense {
	for _, convLayer := range enc.convLayers {
		// Forward pass through convolutional layer
		input = convLayer.Forward(input)
	}
	// Forward pass through pooling layer (there should only ever be 1)
	for _, poolLayer := range enc.poolLayers {
		for i, input_feat := range input {
			input[i] = poolLayer.Forward(input_feat)
		}
	}

	return input
}

// Backward performs a backward pass through the Encoder
func (enc *Encoder) Backward(gradOutput *mat64.Dense, learningRate float64) {
	// Backward pass through convolutional layers
	for i := len(enc.convLayers) - 1; i >= 0; i-- {
		enc.convLayers[i].Backward(gradOutput, learningRate)
	}
}

// Summary returns a summary of the Encoder
func (enc *Encoder) Summary() string {
	summary := "Encoder:\n"
	for i, convLayer := range enc.convLayers {
		summary += fmt.Sprintf("  ConvLayer %d:\n", i)
		summary += convLayer.Summary()
	}
	for i, poolLayer := range enc.poolLayers {
		summary += fmt.Sprintf("  PoolLayer %d:\n", i)
		summary += poolLayer.Summary()
	}
	fmt.Println(summary)
	return summary
}
