package unetTools

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// DecoderParams represents the parameters for a decoder
type DecoderParams struct {
	ConvParams      []ConvParams
	ConvTransParams []ConvTransParams
}

// Decoder represents a decoder for convolutional neural networks
type Decoder struct {
	convParams     []ConvParams
	upsampleParams []ConvTransParams
	convLayers     []*ConvLayer
	upsampleLayers []*ConvTransLayer

	// internal params
	_dWeights []*mat64.Dense
	_dBiases  []*mat64.Dense
}

// NewDecoder initializes a new instance of Decoder
func NewDecoder(convParams []ConvParams, upsampleParams []ConvTransParams) *Decoder {
	decoder := &Decoder{}

	// Create upsampling layers
	for _, params := range upsampleParams {
		decoder.upsampleLayers = append(decoder.upsampleLayers, NewConvTransLayer(params.InputChannels, params.KernelSize, params.Stride, params.NumFilters, params.Activation))
	}

	// Create convolutional layers
	for _, params := range convParams {
		decoder.convLayers = append(decoder.convLayers, NewConvLayer(params.InputChannels, params.KernelSize, params.NumFilters, params.Activation))
	}

	decoder.convParams = convParams
	decoder.upsampleParams = upsampleParams

	decoder._dWeights = make([]*mat64.Dense, len(convParams))
	decoder._dBiases = make([]*mat64.Dense, len(convParams))

	return decoder
}

// Forward performs a forward pass through the Decoder
func (dec *Decoder) Forward(input []*mat64.Dense, skip_features []*mat64.Dense) []*mat64.Dense {
	// upsample input
	for _, upsampleLayer := range dec.upsampleLayers {
		// Forward pass through upsampling layer
		input = upsampleLayer.Forward(input)
	}

	// concatenate with skip features
	// if there are no skip features, then just return the output
	if skip_features != nil {
		// resize skip_features to have the same size as the output
		rows, cols := input[0].Dims()
		for i, skip_feature := range skip_features {
			skip_features[i] = ResizeMatrix(skip_feature, rows, cols)
		}

		// concatenate the output with the skip_features
		input = append(input, skip_features...)
	}
	// pass through convolutional layers
	for _, conv := range dec.convLayers {
		input = conv.Forward(input)
	}
	return input
}

// Backward performs a backward pass through the Decoder
func (dec *Decoder) Backward(outputGrad *mat64.Dense, learningRate float64) {
	// Declare variables for accumulating gradients
	for i := len(dec.convLayers) - 1; i >= 0; i-- {
		// Backward pass through convolutional layer
		dec.convLayers[i].Backward(outputGrad, learningRate)
	}
	for i := len(dec.upsampleLayers) - 1; i >= 0; i-- {
		// Backward pass through upsampling layer
		dec.upsampleLayers[i].Backward(outputGrad, learningRate)
	}
}

// Summary prints a summary of the Decoder
func (dec *Decoder) Summary() string {
	summary := "Decoder:\n"
	for i, cl := range dec.convLayers {
		summary += fmt.Sprintf("  ConvLayer %d:\n", i)
		summary += cl.Summary()
	}
	for i, ul := range dec.upsampleLayers {
		summary += fmt.Sprintf("  UpsampleLayer %d:\n", i)
		summary += ul.Summary()
	}
	fmt.Println(summary)
	return summary
}
