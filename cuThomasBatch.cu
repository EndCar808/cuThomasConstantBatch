// Enda Carroll
// Sept 2019
// Function declarations for cuThomasBatch routine to solve batches of tridiagonal systems

//   Copyright 2019 Enda Carroll

//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

#include "cuThomasBatch.h"



/**
* Function to perform a prefactorization of the LHS using the Thomas algorithm (performed on host device)
*
* @param la Lower diagonal of the LHS matrix - array of lenght n
* @param lb Main diagonal of the LHS matrix - array of lenght n
* * @param lc Upper diagonal of the LHS matrix - array of lenght n
* @param n  Size of the system being solved
*/
void thomasFactorConstantBatch(double* la, double* lb, double* lc, int n) {

	int rowCurrent;
	int rowPrevious;

	rowCurrent = 0;

	// First row
	lb[rowCurrent] = lb[rowCurrent];
	lc[rowCurrent] = lc[rowCurrent] / lb[rowCurrent];

	for (int i = 1; i < n - 1; ++i)	{
		rowPrevious = rowCurrent;
		rowCurrent  += 1;

		la[rowCurrent] = la[rowCurrent];
		lb[rowCurrent] = lb[rowCurrent] - la[rowCurrent]*lc[rowPrevious];
		lc[rowCurrent] = lc[rowCurrent] / lb[rowCurrent];
	}

	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Last row
	la[rowCurrent] = la[rowCurrent];
	lb[rowCurrent] = lb[rowCurrent] - la[rowCurrent]*lc[rowPrevious];
}

/**
* Kernel to solve a prefactorized system using the Thomas alogrithm
* 
* @param la Lower diagonal of the LHS matrix - array of lenght n
* @param lb Main diagonal of the LHS matrix - array of lenght n
* @param lc Upper diagonal of the LHS matrix - array of lenght n
* @param d  RHS array - size n by m
* @param n  Size of the system being solved
* @param m  Size of the batch 
*/
__global__ void cuThomasBatch(double* la, double* lb, double* lc, double* d, int n, int m ) {

	int rowCurrent;
	int rowPrevious;

	int rowAhead;

	// set the current row
	rowCurrent = threadIdx.x + blockDim.x*blockIdx.x;

	int i = 0;

	if ( rowCurrent < m ) 
	{

		//----- Forward Sweep
		d[rowCurrent] = d[rowCurrent] / lb[i];

		#pragma unroll
		for (i = 1; i < n; ++i) {
			rowPrevious = rowCurrent;
			rowCurrent += m;

			d[rowCurrent] = (d[rowCurrent] - la[i]*d[rowPrevious]) / (lb[i]);
		
		}


		//----- Back Sub
		d[rowCurrent] = d[rowCurrent];

		#pragma unroll
		for (i = n - 2; i >= 0; --i) {
			rowAhead    = rowCurrent;
			rowCurrent -= m;

			d[rowCurrent] = d[rowCurrent] - lc[i] * d[rowAhead];
		}
	}
}