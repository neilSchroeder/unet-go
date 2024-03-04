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
	inputSize        int     // Size of input (assumed to be square)
	inputChannels    int     // Number of input channels
	numEnDecoders    int     // Number of encoder-decoder pairs
	numFiltersLayer1 int     // Maximum number of filters in conv layers
	activation       string  // Activation function
	kernelSize       int     // Size of convolutional kernel
	learningRate     float64 // Learning rate
	lossTolerance    float64 // Loss tolerance (will exit if loss less than this value)
	maxIterations    int     // Maximum number of iterations

	lossFunc func(*mat64.Dense, *mat64.Dense) float64 // Loss function

	//internal params
	_steps int
	_loss  float64
	_stop  bool

	encoders   []*Encoder
	bottleneck *Decoder
	decoders   []*Decoder
}

// NewUnet initializes a new instance of Unet
func NewUnet(
	inputSize int,
	inputChannels int,
	numEnDecoders int,
	numFiltersLayer1 int,
	activation string,
	kernelSize int,
	lossFunc func(*mat64.Dense, *mat64.Dense) float64,
) *Unet {

	cl_params := ConvParams{
		activation,
		inputChannels,
		kernelSize,
		numFiltersLayer1,
		0.9,
		0.999,
		1e-8,
	}

	unet := &Unet{
		inputSize:        inputSize,
		inputChannels:    inputChannels,
		numEnDecoders:    numEnDecoders,
		numFiltersLayer1: numFiltersLayer1,
		activation:       activation,
		kernelSize:       kernelSize,
		lossFunc:         lossFunc,

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

	unet._steps = 0
	unet._loss = math.Inf(1) // positive infinity
	unet._stop = false

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
	input = unet.bottleneck.Forward(input, encoder_output[len(encoder_output)-1])

	// pass through decoders
	for i := 0; i < unet.numEnDecoders; i++ {
		input = unet.decoders[i].Forward(input, encoder_output[i])
	}
	// no return, pass by reference
}

// Backward performs a backward pass through the U-Net model
func (unet *Unet) Backward(input *mat64.Dense, target *mat64.Dense, loss float64) (*mat64.Dense, *mat64.Dense, *mat64.Dense) {
	// compute gradients for both weights and biases
	gradOutput := mat64.NewDense(target.RawMatrix().Rows, target.RawMatrix().Cols, nil)
	gradOutput.Sub(target, input)
	gradOutput.Scale(loss, gradOutput)

	rows := unet.encoders[0].convLayers[0].Weights.RawMatrix().Rows
	cols := unet.encoders[0].convLayers[0].Weights.RawMatrix().Cols
	accWeights := mat64.NewDense(rows, cols, nil)
	accBiases := mat64.NewDense(1, cols, nil)
	for _, decode := range unet.decoders {
		w, b := decode.Backward(gradOutput)
		accWeights.Add(accWeights, w)
		accBiases.Add(accBiases, b)
	}
	w, b := unet.bottleneck.Backward(gradOutput)
	accWeights.Add(accWeights, w)
	accBiases.Add(accBiases, b)
	for _, encode := range unet.encoders {
		w, b := encode.Backward(gradOutput)
		accWeights.Add(accWeights, w)
		accBiases.Add(accBiases, b)
	}

	return gradOutput, accWeights, accBiases
}

// Update updates the weights and biases of the U-Net model
func (unet *Unet) Update(weights *mat64.Dense, biases *mat64.Dense, learningRate float64) {
	// update weights and biases
	for _, encode := range unet.encoders {
		encode.Update(weights, biases, learningRate)
	}
	unet.bottleneck.Update(weights, biases, learningRate)
	for _, decode := range unet.decoders {
		decode.Update(weights, biases, learningRate)
	}
}

// Step performs a forward and backward pass through the U-Net model
// followed by an update call
func (unet *Unet) Step(
	input *mat64.Dense,
	target *mat64.Dense,
	learningRate float64,
) float64 {
	unet.Forward(input)
	// compute loss
	unet._loss = unet.lossFunc(input, target)
	_, accWeights, accBiases := unet.Backward(input, target, unet._loss)
	unet.Update(accWeights, accBiases, learningRate)
	unet._steps++
	if unet._steps > unet.maxIterations || unet._loss < unet.lossTolerance {
		unet._stop = true
	}
	/*
		This would be the place to add a check for convergence,
		update the learning rate, etc.
	*/
	return unet._loss
}
