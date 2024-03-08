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
				if matrix.At(i, j) < 0.5 {
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
	ret.Scale(1/255.0, ret)

	return ret
}

// ResizeMatrix resizes a matrix using bilinear interpolation
func ResizeMatrix(input *mat.Dense, newRows, newCols int) *mat.Dense {
	// Get the dimensions of the input matrix
	inputRows, inputCols := input.Dims()

	// Create a new matrix with the desired size
	output := mat.NewDense(newRows, newCols, nil)

	// Calculate the scaling factors
	scaleRow := float64(inputRows-1) / float64(newRows-1)
	scaleCol := float64(inputCols-1) / float64(newCols-1)

	// Perform bilinear interpolation
	for i := 0; i < newRows; i++ {
		for j := 0; j < newCols; j++ {
			// Calculate the corresponding coordinates in the input matrix
			x := float64(i) * scaleRow
			y := float64(j) * scaleCol

			// Find the four nearest neighbors in the input matrix
			x1 := int(x)
			y1 := int(y)
			x2 := x1 + 1
			y2 := y1 + 1

			// Check if the nearest neighbors are within the bounds of the input matrix
			if x2 >= inputRows {
				x2 = x1
			}
			if y2 >= inputCols {
				y2 = y1
			}

			// Perform bilinear interpolation
			dx := x - float64(x1)
			dy := y - float64(y1)
			f11 := input.At(x1, y1)
			f12 := input.At(x1, y2)
			f21 := input.At(x2, y1)
			f22 := input.At(x2, y2)
			interpolated := f11*(1-dx)*(1-dy) + f12*(1-dx)*dy + f21*dx*(1-dy) + f22*dx*dy

			// Set the interpolated value in the output matrix
			output.Set(i, j, interpolated)
		}
	}

	return output
}

// ConcatenateHorizontally concatenates two matrices horizontally
func ConcatenateHorizontally(matrix1, matrix2 *mat.Dense) *mat.Dense {
	// Get the dimensions of the input matrices
	rows1, cols1 := matrix1.Dims()
	rows2, cols2 := matrix2.Dims()

	// Check if the number of rows of both matrices match
	if rows1 != rows2 {
		panic("Number of rows must match for horizontal concatenation")
	}

	// Create the concatenated matrix
	concatenated := mat.NewDense(rows1, cols1+cols2, nil)

	// Copy elements from the first matrix
	concatenated.Augment(matrix1, matrix2)

	return concatenated
}

// ConcatenateVertically concatenates two matrices vertically
func ConcatenateVertically(matrix1, matrix2 *mat.Dense) *mat.Dense {
	// Get the dimensions of the input matrices
	rows1, cols1 := matrix1.Dims()
	rows2, cols2 := matrix2.Dims()

	// Check if the number of columns of both matrices match
	if cols1 != cols2 {
		panic("Number of columns must match for vertical concatenation")
	}

	// Create the concatenated matrix
	concatenated := mat.NewDense(rows1+rows2, cols1, nil)

	// Copy elements from the first matrix
	concatenated.Augment(matrix1, matrix2)

	return concatenated
}

// ConcatenateMatrices concatenates two matrices
func ConcatenateMatrices(matrix1, matrix2 *mat.Dense, axis int) *mat.Dense {
	if axis == 0 {
		return ConcatenateVertically(matrix1, matrix2)
	}
	return ConcatenateHorizontally(matrix1, matrix2)
}
