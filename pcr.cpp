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
			val_j = *(vals + j);
			for (long k = j + 1; k <= end; ++k) {
				//cout << j << "," << k << endl;
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

double* comp_m(const mat_t& U, const mat_t& V, SparseMat* X, int r) {
	long d1 = (*X).d1;
	long d2 = (*X).d2;
	long nnz = (*X).nnz;
	double* m = new double[nnz];
	fill(m, m + nnz, 0.0);
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


mat_t obtain_g(const mat_t& U, const mat_t& V, SparseMat* X, double* m, double lambda) {
	// g is d2 by r, same as V
	// seems faster to move it outside and then pass by reference
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
		fill(t, t + len, 0.0);
		for (long j = start; j <= end - 1; ++j) {
            val_j = *(vals + j);
			for (long k = j + 1; k <= end; ++k) {
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
			update_mat_add_vec(U[i], c, j, g);
		}
		
		delete[] t;
	}
	t = nullptr;
	return g;
}


vec_t compute_Ha(const vec_t& a, double* m, const mat_t& U, SparseMat* X, 
				int r, double lambda) {
	// compute Hessian vector product without explicitly calcualte Hessian H
	// Ha = lambda * a already
	vec_t Ha = copy_vec_t(a, lambda);
	long d1 = X->d1;
	double* vals = X->vals;
	long* rows = X->rows;
	long* index = X->index;
	long start, end, len;	// start, end, denotes starting/ending index in indexs array for i-th user
	long a_start, a_end;	
// a_start, a_end denotes starting/ending index in array 'a' for i-th user, a is size of d2*r
	long mm_start, mm_end;
	long Ha_start, Ha_end;
	double val_j, val_k;
	double y_ijk, mask, ddd;
	double* b;
	double* cpvals;
	r = static_cast<long>(r);	

	for (long i = 0; i < d1; ++i) {
		start = *(index + i);
		end = *(index + i + 1) - 1;
		len = end - start + 1;
		b = new double[len];  // to precompute ui*a
		long cc = 0;
		for (long k = 0; k < len; ++k) {
			long q = *(rows + start + k);
			a_start = q * r;
			a_end = (q + 1) * r - 1;
			b[cc++] = vec_prod_array(U[i], a, a_start, a_end);
		}
		cpvals = new double[len];
		fill(cpvals, cpvals + len, 0.0);
		for (long j = start; j < end; ++j) {
			val_j = *(vals + j);

			for (long k = j + 1; k <= end; ++k) {
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
                    ddd = *(b + j - start) - *(b + k - start);
					ddd *= 2.0;

                    *(cpvals + j - start) += ddd;
                    *(cpvals + k - start) -= ddd;
                }
			}
		}
		for (long k = 0; k < len; ++k) {
            long p = *(rows + start + k);
            double c = *(cpvals + k);
			Ha_start = p * r;
			Ha_end = (p + 1) * r - 1;
			update_vec_subrange(U[i], c, Ha, Ha_start, Ha_end);	
        }	
		delete[] b;
		delete[] cpvals;
	}
	b = nullptr;
	cpvals = nullptr;
	return Ha;
}




vec_t solve_delta(const vec_t& g, double* m, const mat_t& U, SparseMat* X, int r, 
				double lambda) {
	vec_t delta = vec_t(g.size(), 0.0);
	vec_t rr = copy_vec_t(g, -1.0);
	vec_t p = copy_vec_t(rr, -1.0);
	double err = sqrt(norm(rr)) * 0.01;
	cout << "break condition " << err << endl;
	for (int k = 1; k <= 30; ++k) {
		//vec_t Hp = copy_vec_t(p, lambda);
		vec_t Hp = compute_Ha(p, m, U, X, r, lambda);

		double prod_p_Hp = dot(p, Hp);
		
		double alpha = -1.0 * dot(rr, p) / prod_p_Hp;
		
		delta = add_vec_vec(delta, p, 1.0, alpha);
		rr = add_vec_vec(rr, Hp, 1.0, alpha);
		cout << "In CG, iteration " << k << " rr value:" << sqrt(norm(rr)) << endl;
		if (sqrt(norm(rr)) < err) {
			break;
		}
		double b = dot(rr, Hp) / prod_p_Hp;
		p = add_vec_vec(rr, p, -1.0, b);
	}
	return delta;
}

double* update_V(SparseMat* X, double lambda, double stepsize, int r, 
				 mat_t& U, mat_t& V, double& now_obj) {
	// update V while fixing U fixed
	double* m = comp_m(U, V, X, r);
	double time = omp_get_wtime();
	
	//mat_t g = copy_mat_t(V, lambda);
	mat_t g = obtain_g(U, V, X, m, lambda);
	
	cout << "time for obtain_g function takes " << omp_get_wtime() - time << endl;

	cout << "g is succesfully computed " << g.size() << "," << g[0].size() << endl;  
	// vectorize_mat function to convert g from mat_t into vec_t 
	vec_t g_vec;	
	vectorize_mat(g, g_vec);
	cout << norm(g) - norm(g_vec) << endl;
 	//assert(norm(g) == norm(g_vec));	
	cout << "vectorization is successful, now size is " << g_vec.size() << endl;
	
	// solve_delta function to implement conjugate gradient algorithm
	vec_t delta = solve_delta(g_vec, m, U, X, r, lambda);
	
	// reshape function (not needed if implement mat_t substract vec_t function)
	
	cout << "solve_delta is okay" << endl;	
	
	double prev_obj = objective(m, U, V, X, lambda);
	
	mat_t V_old;
	//V_old = copy_mat_t(V, 1.0);
	cout << "stepsize is " << stepsize << endl;
	cout << "norm of delta is " << norm(delta) << endl;

	// truncated newton till convergence
	for (int iter = 0; iter < 20; ++iter) {
		V_old = copy_mat_t(V, 1.0);

		mat_substract_vec(delta, stepsize, V_old);
		
		delete[] m;
		//cout << V_old[0][0] << endl;
		//cout << V[0][0] << endl;
		m = comp_m(U, V_old, X, r);
		
		now_obj = objective(m, U, V_old, X, lambda);

		cout << "Line Search Iter " << iter << " Prev Obj " << prev_obj 
			 << " New Obj" << now_obj << " stepsize " << stepsize << endl;
		
		if (now_obj < prev_obj) {
			V = copy_mat_t(V_old, 1.0);
			break;
		} else {
			stepsize /= 2.0;
		}
	}

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
	cout << "stepsize is " << stepsize << "ndcg_k is" << ndcg_k << endl;
	// X: d1 by d2 sparse matrix, ratings
	// U: r by d1 dense
	// V: r by d2 dense
	// X ~ U^T * V
	
	SparseMat* X = convert(R);  // remember to free memory X in the end by delete X; set X = NULL;
	long nnz = (*X).nnz;
	cout << nnz << endl;
	double* m = comp_m(U, V, X, r);
	cout << "so far so good in pcr"<<endl;
	cout << "First and Last element in m array" << *m << " " << *(m + nnz - 1) << endl;

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
