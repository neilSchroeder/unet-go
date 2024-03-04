package unetTools_test

import (
	"testing"

	"unet-go/unet/internal/pkg/unetTools"

	"github.com/gonum/matrix/mat64"
)

func TestNewConvLayer(t *testing.T) {
	inputChannels := 3
	kernelSize := 3
	numFilters := 16
	activation := "relu"

	cl := unetTools.NewConvLayer(inputChannels, kernelSize, numFilters, activation)

	if cl.InputChannels != inputChannels {
		t.Errorf("Expected InputChannels to be %d, but got %d", inputChannels, cl.InputChannels)
	}

	if cl.KernelSize != kernelSize {
		t.Errorf("Expected KernelSize to be %d, but got %d", kernelSize, cl.KernelSize)
	}

	if cl.NumFilters != numFilters {
		t.Errorf("Expected NumFilters to be %d, but got %d", numFilters, cl.NumFilters)
	}

	if cl.Activation != activation {
		t.Errorf("Expected Activation to be %s, but got %s", activation, cl.Activation)
	}
}

func TestConvLayerForward(t *testing.T) {
	inputChannels := 3
	kernelSize := 3
	numFilters := 16
	activation := "relu"

	cl := unetTools.NewConvLayer(inputChannels, kernelSize, numFilters, activation)

	input := mat64.NewDense(10, 10, nil)

	_ = cl.Forward(input)

	// Add assertions for the expected output

}

func TestConvLayerBackward(t *testing.T) {
	inputChannels := 3
	kernelSize := 3
	numFilters := 16
	activation := "relu"

	cl := unetTools.NewConvLayer(inputChannels, kernelSize, numFilters, activation)

	gradOutput := mat64.NewDense(10, 10, nil)

	_, _ = cl.Backward(gradOutput)

	// Add assertions for the expected gradInput and gradWeights
}

func TestConvLayerUpdateWeightsAndBiases(t *testing.T) {
	inputChannels := 3
	kernelSize := 3
	numFilters := 16
	activation := "relu"

	cl := unetTools.NewConvLayer(inputChannels, kernelSize, numFilters, activation)

	learningRate := 0.001
	gradWeights := mat64.NewDense(10, 10, nil)
	gradBiases := mat64.NewDense(1, numFilters, nil)

	cl.UpdateWeightsAndBiases(learningRate, gradWeights, gradBiases)

	// Add assertions for the updated Weights and biases
}
