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

	return encoder
}

// Forward performs a forward pass through the Encoder
func (enc *Encoder) Forward(input *mat64.Dense) *mat64.Dense {
	output := input
	for _, convLayer := range enc.convLayers {
		// Forward pass through convolutional layer
		output = convLayer.Forward(output)
	}
	// Forward pass through pooling layer (there should only ever be 1)
	for _, poolLayer := range enc.poolLayers {
		output = poolLayer.Forward(output)
	}

	return output
}

// Backward performs a backward pass through the Encoder
func (enc *Encoder) Backward(gradOutput *mat64.Dense) (*mat64.Dense, *mat64.Dense) {
	// Declare variables for accumulating gradients
	rows := enc.convLayers[0].Weights.RawMatrix().Rows
	cols := enc.convLayers[0].Weights.RawMatrix().Cols
	accWeights := mat64.NewDense(rows, cols, nil)
	accBiases := mat64.NewDense(1, cols, nil)

	// Backward pass through pooling layer (there should only ever be 1)
	for i := len(enc.poolLayers) - 1; i >= 0; i-- {
		gradOutput = enc.poolLayers[i].Backward(gradOutput)
	}

	// Backward pass through convolutional layers
	for i := len(enc.convLayers) - 1; i >= 0; i-- {
		weights, biases := enc.convLayers[i].Backward(gradOutput)

		// Accumulate gradients
		accWeights.Add(accWeights, weights)
		accBiases.Add(accBiases, biases)
	}

	return accWeights, accBiases
}

// Update updates the weights and biases of the Encoder
func (enc *Encoder) Update(weights *mat64.Dense, biases *mat64.Dense, learningRate float64) {
	for _, convLayer := range enc.convLayers {
		convLayer.UpdateWeightsAndBiases(learningRate, weights, biases)
	}
}
