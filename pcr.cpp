#include "util.h"
#include "pmf.h"


double objective(double* m, mat_t& U, mat_t& V, SparseMat* X, double lambda) {
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

double* comp_m(mat_t& U, mat_t& V, SparseMat* X, int r) {
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


mat_t obtain_g(mat_t& U, mat_t& V, SparseMat* X, double* m, double lambda) {
	// g is d2 by r, same as V
	mat_t g = copy_mat_t(V, lambda);  // g=lambda*V
	long d1 = X->d1;
	double* vals = X->vals;
    long* index = X->index;
	long* rows = X->rows;

    long start, end, len;
    double val_j, val_k;
    double y_ijk, mask, s_jk;	
	double* t;
	
	for (long i = 0; i < d1; ++i) {
		start = *(index + i);
        end = *(index + i + 1) - 1;
		len = end - start + 1;
		t = new double[len]; 	// t is pointer to array length of len
		for (long j = start; j <= end - 1; ++j) {
            for (long k = j + 1; k <= end; ++k) {
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
					s_jk = 2.0 * (mask - 1);
					*(t + j - start) += s_jk;
					*(t + k - start) -= s_jk;
				}
			}
		}
		for (long k = 0; k < len; ++k) {
			long j = *(rows + start + k);
			double c = *(t + k); 
			// we want g[j,:] += c * U[i,:]
			update_mat_add_vec(U, i, c, j, g);
		}
		
		delete[] t;
	}
	t = nullptr;
	return g;
}

double* update_V(SparseMat* X, double lambda, double stepsize, int r, mat_t& U, mat_t& V, double& now_obj) {
	double* m = comp_m(U, V, X, r);
	double time = omp_get_wtime();
	
	mat_t g = obtain_g(U, V, X, m, lambda);
	cout << "time for obtain_g function takes " << omp_get_wtime() - time << endl;

	cout << "g is succesfully computed " << g.size() << "," << g[0].size() << endl;  


	return m;
}



// Primal-CR Algorithm
void pcr(smat_t& R, mat_t& U, mat_t& V, parameter& param) {
	cout<<"enter pcr"<<endl;
	int r = param.k;
	double lambda = param.lambda;
	double stepsize = param.stepsize;
	int ndcg_k = param.ndcg_k;
	double now_obj;
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
	now_obj = objective(m, U, V, X, lambda);
	
	cout << "time for objective function takes " << omp_get_wtime() - time << endl;
	cout << "iniitial objective is " << now_obj << endl;
	
	int num_iter = 1;
	
	
	/*
	mat_t g = copy_mat_t(V);
	
	cout << g.size() << ", g, " << g[0].size() << endl;
	for (long i = 0; i < X->d2; ++i) {
		cout << g[i][0] << endl;
	}
	*/
	for (int iter = 0; iter < num_iter; ++iter) {
		// need to free space pointer m points to before pointing it to another memory
		delete[] m;
		m = update_V(X, lambda, stepsize, r, U, V, now_obj);
	}
	
	
	
	
	
	delete[] m;
	delete X;
	//m = NULL;
	//X = NULL;
	m = nullptr;
	X = nullptr;
	return;
}
