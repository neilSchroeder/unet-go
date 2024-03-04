package unetTools

import (
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"math/rand"
	"os"

	"github.com/gonum/matrix/mat64"
	mat "github.com/gonum/matrix/mat64"
)

// Apply Activation function to the input matrix
func applyActivation(matrix *mat64.Dense, activation string) {
	numRows, numCols := matrix.Dims()
	switch activation {
	case "relu":
		for i := 0; i < numRows; i++ {
			for j := 0; j < numCols; j++ {
				if matrix.At(i, j) < 0 {
					matrix.Set(i, j, 0)
				}
			}
		}
	case "sigmoid":
		for i := 0; i < numRows; i++ {
			for j := 0; j < numCols; j++ {
				matrix.Set(i, j, 1/(1+math.Exp(-matrix.At(i, j))))
			}
		}
		// Add more activation functions as needed
	}
}

// randomMatrixValues generates random values for a matrix of the specified size
func randomMatrixValues(size int) []float64 {
	values := make([]float64, size)
	for i := range values {
		values[i] = rand.Float64()
	}
	return values
}

// MatrixSqrt computes the square root of each element in the input matrix
func MatrixSqrt(matrix *mat64.Dense) *mat64.Dense {
	numRows, numCols := matrix.Dims()
	output := mat64.NewDense(numRows, numCols, nil)
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			output.Set(i, j, math.Sqrt(matrix.At(i, j)))
		}
	}
	return output
}

// MatrixAddConst adds a constant to each element in the input matrix
func MatrixAddConst(matrix *mat64.Dense, constant float64) *mat64.Dense {
	numRows, numCols := matrix.Dims()
	output := mat64.NewDense(numRows, numCols, nil)
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			output.Set(i, j, matrix.At(i, j)+constant)
		}
	}
	return output
}

func MatrixDivConst(matrix *mat64.Dense, constant float64) *mat64.Dense {
	numRows, numCols := matrix.Dims()
	output := mat64.NewDense(numRows, numCols, nil)
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			output.Set(i, j, matrix.At(i, j)/constant)
		}
	}
	return output
}

func ConstDivMatrix(constant float64, matrix *mat64.Dense) *mat64.Dense {
	numRows, numCols := matrix.Dims()
	output := mat64.NewDense(numRows, numCols, nil)
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			output.Set(i, j, constant/matrix.At(i, j))
		}
	}
	return output
}

// LoadImage takes in a file path and returns a matrix representation of the image
func LoadImage(filePath string) *mat64.Dense {
	// Open the JPEG file
	var ret *mat64.Dense
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Println("Error:", err)
		return ret
	}
	defer file.Close()

	// Decode the JPEG image
	img, err := jpeg.Decode(file)
	if err != nil {
		fmt.Println("Error:", err)
		return ret
	}

	// Convert the image to grayscale
	grayImg := image.NewGray(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			grayImg.Set(x, y, img.At(x, y))
		}
	}

	// Convert the grayscale image to a matrix
	rows, cols := grayImg.Bounds().Max.Y, grayImg.Bounds().Max.X
	ret = mat.NewDense(rows, cols, nil)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			pixel := grayImg.GrayAt(x, y)
			ret.Set(y, x, float64(pixel.Y))
		}
	}

	return ret
}
