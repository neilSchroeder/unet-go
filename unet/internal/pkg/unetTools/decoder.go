package unetTools

import (
	"slices"

	"github.com/gonum/matrix/mat64"
)

// DecoderParams represents the parameters for a decoder
type DecoderParams struct {
	ConvParams     []ConvParams
	UpsampleParams []UpsampleParams
}

// Decoder represents a decoder for convolutional neural networks
type Decoder struct {
	convParams     []ConvParams
	upsampleParams []UpsampleParams
	convLayers     []*ConvLayer
	upsampleLayers []*UpsampleLayer
}

// NewDecoder initializes a new instance of Decoder
func NewDecoder(convParams []ConvParams, upsampleParams []UpsampleParams) *Decoder {
	decoder := &Decoder{}

	// Create upsampling layers
	for _, params := range upsampleParams {
		decoder.upsampleLayers = append(decoder.upsampleLayers, NewUpsampleLayer(params.ScaleFactor))
	}

	// Create convolutional layers
	for _, params := range convParams {
		decoder.convLayers = append(decoder.convLayers, NewConvLayer(params.InputChannels, params.KernelSize, params.NumFilters, params.Activation))
	}

	decoder.convParams = convParams
	decoder.upsampleParams = upsampleParams

	return decoder
}

// Forward performs a forward pass through the Decoder
func (dec *Decoder) Forward(input *mat64.Dense, skip_features *mat64.Dense) *mat64.Dense {
	output := input
	// upsample input
	for _, upsampleLayer := range dec.upsampleLayers {
		// Forward pass through upsampling layer
		output = upsampleLayer.Forward(output)
	}

	// concatenate with skip features
	outputSlice := output.RawMatrix().Data
	skipFeaturesSlice := skip_features.RawMatrix().Data
	concatenated := slices.Concat(outputSlice, skipFeaturesSlice)
	output = mat64.NewDense(output.RawMatrix().Rows, output.RawMatrix().Cols, concatenated)

	// pass through convolutional layers
	for _, conv := range dec.convLayers {
		// Forward pass through upsampling layer
		output = conv.Forward(output)
	}
	return output
}

// Backward performs a backward pass through the Decoder
func (dec *Decoder) Backward(outputGrad *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	// Declare variables for accumulating gradients
	rows := dec.convParams[0].InputChannels * dec.convParams[0].KernelSize * dec.convParams[0].KernelSize
	cols := dec.convParams[0].NumFilters
	accWeights := mat64.NewDense(rows, cols, nil)
	accBiases := mat64.NewDense(1, cols, nil)
	for _, cl := range dec.convLayers {
		// Backward pass through convolutional layer
		dWeights, dBiases := cl.Backward(outputGrad)
		accWeights.Add(accWeights, dWeights)
		accBiases.Add(accBiases, dBiases)
	}
	return accWeights, accBiases
}

// Update updates the weights and biases of the Decoder using the accumulated
// gradients and the learning rate
func (dec *Decoder) Update(weights *mat64.Dense, biases *mat64.Dense, learningRate float64) {
	for _, cl := range dec.convLayers {
		// Update weights and biases of convolutional layer
		cl.UpdateWeightsAndBiases(learningRate, weights, biases)
	}
}
