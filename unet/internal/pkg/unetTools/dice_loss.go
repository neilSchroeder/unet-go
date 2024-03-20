package unetTools

import (
	"math"

	mat "github.com/gonum/matrix/mat64"
)

// DiceLoss calculates the Dice loss between two binary masks
func DiceLoss(prediction, target *mat.Dense) float64 {
	// Convert prediction and target masks to slices
	pred := prediction.RawMatrix().Data
	targ := target.RawMatrix().Data

	// Initialize variables for intersection and union
	intersection := 0.0
	union := 0.0

	// Calculate intersection and union
	for i := 0; i < len(pred); i++ {
		if pred[i] >= 0.9 && targ[i] == 1 {
			intersection++
		}
		if pred[i] >= 0.9 || targ[i] == 1 {
			union++
		}
	}

	// Calculate Dice coefficient
	dice := 2.0 * intersection / (union + intersection)

	// Calculate Dice loss
	diceLoss := 1.0 - dice

	return diceLoss
}

// DiceLossGradient calculates the gradient of the Dice loss between two binary masks
func DiceLossGradient(prediction, target *mat.Dense) *mat.Dense {
	// Compute sums of prediction and target masks
	p := mat.Sum(target)
	q := mat.Sum(prediction)

	// Compute the gradient of the Dice Loss
	gradient := mat.NewDense(target.RawMatrix().Rows, prediction.RawMatrix().Cols, nil)
	gradient.Apply(func(_, _ int, v float64) float64 {
		return -2 * v / math.Pow(p+q, 2)
	}, target)

	return gradient
}
