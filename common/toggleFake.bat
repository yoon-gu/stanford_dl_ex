@echo off

rem use the smaller test data set for training during debugging

if exist train-images-idx3-ubyte-fake (
	rename train-images-idx3-ubyte train-images-idx3-ubyte-real
	rename train-images-idx3-ubyte-fake train-images-idx3-ubyte

	rename train-labels-idx1-ubyte train-labels-idx1-ubyte-real
	rename train-labels-idx1-ubyte-fake train-labels-idx1-ubyte
) else (
	rename train-images-idx3-ubyte train-images-idx3-ubyte-fake
	rename train-images-idx3-ubyte-real train-images-idx3-ubyte

	rename train-labels-idx1-ubyte train-labels-idx1-ubyte-fake 
	rename train-labels-idx1-ubyte-real train-labels-idx1-ubyte
)