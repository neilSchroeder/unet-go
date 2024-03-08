package unetTools

import (
	"fmt"
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
	poolSize         int     // Size of pooling kernel
	poolStride       int     // Stride of pooling kernel
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
	poolSize int,
	poolStride int,
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
	ctl_params := ConvTransParams{
		activation,
		inputChannels,
		kernelSize,
		poolStride,
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
		poolSize:         poolSize,
		poolStride:       poolStride,
		lossFunc:         lossFunc,

		encoders: make([]*Encoder, numEnDecoders),
		bottleneck: NewDecoder(
			[]ConvParams{cl_params, cl_params},
			[]ConvTransParams{ctl_params},
		),
		decoders: make([]*Decoder, numEnDecoders),
	}

	// build the encoder-decoder pairs
	for i := 0; i < numEnDecoders; i++ {
		unet.encoders[i] = NewEncoder(
			[]ConvParams{cl_params, cl_params},
			[]PoolParams{{poolSize, poolStride}},
		)
		cl_params.NumFilters *= 2

	}
	unet.bottleneck = NewDecoder(
		[]ConvParams{cl_params, cl_params},
		[]ConvTransParams{},
	)
	for i := 0; i < numEnDecoders; i++ {
		cl_params_half := cl_params
		cl_params_half.NumFilters /= 2
		unet.decoders[i] = NewDecoder(
			[]ConvParams{cl_params, cl_params_half},
			[]ConvTransParams{ctl_params},
		)
		cl_params.NumFilters /= 2
	}

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
	slices.Reverse(encoder_output)

	// handle bottleneck
	// bottleneck doesn't have a skip connection
	input = unet.bottleneck.Forward(input, nil)

	// pass through decoders
	for i := 0; i < unet.numEnDecoders; i++ {
		input = unet.decoders[i].Forward(input, encoder_output[i])
	}
	// no return, pass by reference
}

// Backward performs a backward pass through the U-Net model
func (unet *Unet) Backward(input *mat64.Dense, target *mat64.Dense, loss float64) {
	// compute gradients for both weights and biases
	gradOutput := mat64.NewDense(target.RawMatrix().Rows, target.RawMatrix().Cols, nil)
	gradOutput.Sub(target, input)
	gradOutput.Scale(loss, gradOutput)

	for _, decode := range unet.decoders {
		decode.Backward(gradOutput)
	}
	unet.bottleneck.Backward(gradOutput)
	for _, encode := range unet.encoders {
		encode.Backward(gradOutput)
	}
}

// Update updates the weights and biases of the U-Net model
func (unet *Unet) Update() {
	// update weights and biases
	for _, encode := range unet.encoders {
		encode.Update(unet.learningRate)
	}
	unet.bottleneck.Update(unet.learningRate)
	for _, decode := range unet.decoders {
		decode.Update(unet.learningRate)
	}
}

// Step performs a forward and backward pass through the U-Net model
// followed by an update call
func (unet *Unet) Step(
	input *mat64.Dense,
	target *mat64.Dense,
	learningRate float64,
) float64 {
	fmt.Println("[INFO] UNet Forward:")
	unet.Forward(input)
	// compute loss
	unet._loss = unet.lossFunc(input, target)
	fmt.Println("[INFO] UNet Loss:", unet._loss)
	fmt.Println("[INFO] UNet Backward:")
	unet.Backward(input, target, unet._loss)
	unet.Update()
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

// Summary returns a string representation of the U-Net model
func (unet *Unet) Summary() string {
	// print summary of each encoder
	for _, encode := range unet.encoders {
		encode.Summary()
	}
	unet.bottleneck.Summary()
	for _, decode := range unet.decoders {
		decode.Summary()
	}
	return "Unet"
}

// GetLoss returns the current loss of the U-Net model
func (unet *Unet) GetLoss() float64 {
	return unet._loss
}
