# nla_challenges

### Task3: 
- The euclidean norm of v is calculated from the normalized vector of v.
- w size: 87296, v size: 87296.

### Task4: 
- If an element with a certain index does not exist, we simply assume the corresponding term to be zero.
- i-th row of A1 corresponds to applying the smoothing filter matrix to the i-th pixel. Each pixel interacts only with max nine pixel that are also the nnz of the row. We conclude that the matrix is sparse.

### Task6:
- Constructed similar to A1 but there are max five nnz elements per row.

### Task8:
- We used the folder test of lis library to compute the results.
- We concluded that GMRES was the fastest method shown in the lectures.
- We compared the method with ILU preconditioner(24 iterations) and without preconditioner(93 iterations). The fastest preconditioner we tested was ILUT(11 iterations) but it wasn't shown in the course.

### Task10:
- see task 6

### Task12:
- We used the BiCG method without the preconditioner and we obtained:
	Eigen native BiCG
	#iterations:     23
	relative residual: 7.89737e-11
	Computation time: 0.624028 seconds
- The BiCG with ILUT preconditioner has been tested and we obtained:
	Eigen BiCGSTAB  (IncompleteLUT)
	#iterations:     3
	relative residual: 8.16343e-17
	Computation time: 7.4835 seconds
- We observed that the number of iteration decreased with the use of preconditioner but the computation time was roughly ten times bigger.