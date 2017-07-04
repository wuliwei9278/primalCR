#include "util.h"
#include "pmf.h"


double objective(double* m, mat_t& U, mat_t& V, SparseMat* X, double lambda){
	double res = 0;
	double norm_U = norm(U);
	double norm_V = norm(V);
	res += lambda * (norm_U + norm_V) / 2.0;
	long d1 = X->d1;
	double* vals = X->vals;
	long* index = X->index;
	long start, end;
	double val_j, val_k;
	double y_ijk, mask;
//	cout << "enter for loop" << endl;
	for (long i = 0; i < d1; ++i) {
		start = *(index + i);
		end = *(index + i + 1) - 1;
//		cout << i << " " << start << " " << end << endl;
		for (long j = start; j <= end - 1; ++j) {
			for (long k = j + 1; k <= end; ++k) {
				//cout << j << "," << k << endl;
				val_j = *(vals + j);
				val_k = *(vals + k);
				if (val_j == val_k) {
					continue;
				} else if (val_j > val_k) {
					y_ijk = 1.0;	
				} else {
					y_ijk = -1.0;
				}
				mask = *(m + j) - *(m + k);
				mask *= y_ijk;
				if (mask < 1.0) {
					res += (1.0 - mask) * (1.0 - mask);
				}
			}
		}
	}
	return res;
}

double* comp_m(mat_t& U, mat_t& V, SparseMat* X, int r){
	long d1 = (*X).d1;
	long d2 = (*X).d2;
	long nnz = (*X).nnz;
	double* m = new double[nnz];
	long usr_id, item_id;
	double dot_res;
	for (long i = 0; i < nnz; ++i) {
		usr_id = (*X).cols[i];
		item_id = (*X).rows[i];
		dot_res = 0;
		for (int j = 0; j < r; ++j) {
			dot_res = dot_res + U[usr_id][j] * V[item_id][j];
		}
		*(m + i) = dot_res;
	}
	return m;  // remember to free memory of pointer m

}

// Primal-CR Algorithm
void pcr(smat_t& R, mat_t& U, mat_t& V, parameter& param){
	cout<<"enter pcr"<<endl;
	int r = param.k;
	double lambda = param.lambda;
	double stepsize = param.stepsize;
	int ndcg_k = param.ndcg_k;
	double obj = 0.0;
	double totaltime = 0.0;

	// X: d1 by d2 sparse matrix, ratings
	// U: r by d1 dense
	// V: r by d2 dense
	// X ~ U^T * V
	
	SparseMat* X = convert(R);  // remember to free memory X in the end by delete X; set X = NULL;
	long nnz = (*X).nnz;
	cout << nnz << endl;
	double* m = comp_m(U, V, X, r);
	cout << "so far so good in pcr"<<endl;
	cout << *m << " " << *(m + nnz - 1) << endl;

	double time = omp_get_wtime();
	double now_obj = objective(m, U, V, X, lambda);
	cout << "time for objective function takes " << omp_get_wtime() - time << endl;
	cout << "iniitial objective is " << now_obj << endl;
	delete m;
	delete X;
	m = NULL;
	X = NULL;
	return;
}
