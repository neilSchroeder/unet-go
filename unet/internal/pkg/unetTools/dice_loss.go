package unetTools

import (
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
		if pred[i] == 1 && targ[i] == 1 {
			intersection++
		}
		if pred[i] == 1 || targ[i] == 1 {
			union++
		}
	}

	// Calculate Dice coefficient
	dice := 2.0 * intersection / (union + intersection)

	// Calculate Dice loss
	diceLoss := 1.0 - dice

	return diceLoss
}
