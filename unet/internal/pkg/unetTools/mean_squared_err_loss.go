package unetTools

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func MeanSquaredErr(prediction *mat64.Dense, target *mat64.Dense) float64 {
	// Convert prediction and target masks to slices
	pred := prediction.RawMatrix().Data
	targ := target.RawMatrix().Data

	// Initialize variable for mean squared error
	mse := 0.0

	// Calculate mean squared error
	for i := 0; i < len(pred); i++ {
		mse += math.Pow(pred[i]-targ[i], 2)
	}
	mse /= float64(len(pred))

	return mse
}

func MeanSquaredErrGradient(prediction *mat64.Dense, target *mat64.Dense) *mat64.Dense {
	// Compute the gradient of the mean squared error
	gradient := mat64.NewDense(target.RawMatrix().Rows, prediction.RawMatrix().Cols, nil)
	gradient.Sub(prediction, target)

	return gradient
}
