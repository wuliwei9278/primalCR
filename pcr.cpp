#include "util.h"
#include "pmf.h"
#define kind dynamic,500

double objective(double* m, const mat_t& U, const mat_t& V, SparseMat* X, double lambda) {
	double res = 0;
	double norm_U = norm(U);
	double norm_V = norm(V);
	long d1 = X->d1;
	double* vals = X->vals;
	long* index = X->index;

#pragma omp parallel for schedule(kind) reduction(+:res)
	for (long i = 0; i < d1; ++i) {
		long start = *(index + i);
		long end = *(index + i + 1) - 1;
//		cout << i << " " << start << " " << end << endl;
		for (long j = start; j <= end - 1; ++j) {
			double val_j = *(vals + j);
			for (long k = j + 1; k <= end; ++k) {
				//cout << j << "," << k << endl;
				double val_k = *(vals + k);
				if (val_j == val_k) {
					continue;
				} /*else if (val_j > val_k) {
					y_ijk = 1.0;	
				} else {
					y_ijk = -1.0;
				}*/
				double mask = *(m + j) - *(m + k);
//				mask *= y_ijk;
				if ( val_j < val_k )
					mask =-mask;
				if (mask < 1.0) {
					res += (1.0 - mask) * (1.0 - mask);
				}
			}
		}
	}

	res += lambda * (norm_U + norm_V) / 2.0;
	return res;
}



double* comp_m(const mat_t& U, const mat_t& V, SparseMat* X, int r) {
	long d1 = (*X).d1;
	long d2 = (*X).d2;
	long nnz = (*X).nnz;
	double* m = new double[nnz];
	fill(m, m + nnz, 0.0);

#pragma omp parallel for schedule(kind)
	for (long i = 0; i < nnz; ++i) {
		long usr_id = (*X).cols[i];
		long item_id = (*X).rows[i];
		double dot_res = 0;
		for (int j = 0; j < r; ++j) {
			dot_res = dot_res + U[usr_id][j] * V[item_id][j];
		}
		*(m + i) = dot_res;
	}
	return m;  // remember to free memory of pointer m

}

void update_m(long i, const vec_t& ui, const mat_t& V, SparseMat* X, int r, double* m) {
	long* rows = X->rows;
	long* index = X->index;
	long start = *(index + i);
	long end = *(index + i + 1) - 1;
	long len = end - start + 1;

	for (long j = start; j <= end; ++j) {
		long item_id = *(rows + j);
		double dot_res = dot(ui, V[item_id]);
		*(m + j) = dot_res;
	}
	return;
}

double* compute_mm(long i, const vec_t& ui_new, const mat_t& V, SparseMat* X, int r) {
	long* rows = X->rows;
	long* index = X->index;
	long start = *(index + i);
	long end = *(index + i + 1);
	long len = end - start;
	double* mm = new double[len];
	for (long j = start; j < end; ++j) {
		long item_id = *(rows + j);
		double res = 0.0;
		for (int k = 0; k < r; ++k) {
			res += ui_new[k] * V[item_id][k];
		}
		*(mm + j - start) = res;
	}
	return mm;
}


mat_t obtain_g(const mat_t& U, const mat_t& V, SparseMat* X, double* m, double lambda) {
	// g is d2 by r, same as V
	// seems faster to move it outside and then pass by reference
	mat_t g = copy_mat_t(V, lambda);  // g=lambda*V
	long d1 = X->d1;
	double* vals = X->vals;
    	long* index = X->index;
	long* rows = X->rows;
	int r = g[0].size();
	long d2 = g.size();

	int num_threads = omp_get_max_threads();
//	vector<mat_t> g_list(num_threads, mat_t(d2, vec_t(r, 0)));


#pragma omp parallel for schedule(kind) 
	for (long i = 0; i < d1; ++i) {
		int rank = omp_get_thread_num();
		long start = *(index + i);
      	long end = *(index + i + 1) - 1;
		long len = end - start + 1;
		double *t = new double[len]; 	// t is pointer to array length of len
		fill(t, t + len, 0.0);
		for (long j = start; j <= end - 1; ++j) {
            double val_j = *(vals + j);
			for (long k = j + 1; k <= end; ++k) {
                	double val_k = *(vals + k);
					double y_ijk = 1.0;
				if (val_j == val_k) {
                    			continue;
                		} else if (val_j < val_k) {
                    			y_ijk = -1.0;
                		}
				double mask = *(m + j) - *(m + k);
                		mask *= y_ijk;
				if (mask < 1.0) {
					double s_jk = 2.0 * (mask - 1);
					*(t + j - start) += s_jk*y_ijk;
					*(t + k - start) -= s_jk*y_ijk;
				}
			}
		}
		for (long k = 0; k < len; ++k) {
			long j = *(rows + start + k);
			double c = *(t + k); 
			// we want g[j,:] += c * U[i,:]
//			update_mat_add_vec(U[i], c, j, g);
			for ( int k=0 ; k<r ; k++ )
#pragma omp atomic
				g[j][k] += c*U[i][k];
//			update_mat_add_vec(U[i], c, j, g_list[rank]);
		}
		
		delete[] t;
	}
/*		for ( int i=0 ; i<d2 ; i++ )
			for ( int k=0 ; k<r ; k++ )
				for ( int ii=0 ; ii<num_threads ; ii++)
					g[i][k]+= g_list[ii][i][k];
					*/
//	t = nullptr;
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
//	long start, end, len;	// start, end, denotes starting/ending index in indexs array for i-th user
	// a_start, a_end denotes starting/ending index in array 'a' for i-th user, a is size of d2*r
//	long a_start, a_end;	
//	long Ha_start, Ha_end;
//	double val_j, val_k;
//	double mask, ddd;
//	int y_ijk;
//	double* b;
//	double* cpvals;
	r = static_cast<long>(r);	

#pragma omp parallel for schedule(kind) 
	for (long i = 0; i < d1; ++i) {
		long start = *(index + i);
		long end = *(index + i + 1) - 1;
		long len = end - start + 1;
		double *b = new double[len];  // to precompute ui*a
		long cc = 0;
		for (long k = 0; k < len; ++k) {
			long q = *(rows + start + k);
			long a_start = q * r;
			long a_end = (q + 1) * r - 1;
			b[cc++] = vec_prod_array(U[i], a, a_start, a_end);
		}
		double *cpvals = new double[len];
		fill(cpvals, cpvals + len, 0.0);
		for (long j = start; j < end; ++j) {
			double val_j = *(vals + j);

			for (long k = j + 1; k <= end; ++k) {
                double val_k = *(vals + k);
                
				if (val_j == val_k) {
                    continue;
                } 
                
				double mask = *(m + j) - *(m + k);
				if ( val_k > val_j )
					mask = -mask;
                
				if (mask < 1.0) {
                    double ddd = *(b + j - start) - *(b + k - start);
					ddd *= 2;

                    *(cpvals + j - start) += ddd;
                    *(cpvals + k - start) -= ddd;
                }
			}
		}
		for (long k = 0; k < len; ++k) {
            long p = *(rows + start + k);
            double c = *(cpvals + k);
			long Ha_start = p * r;
			long Ha_end = (p + 1) * r - 1;
			//
			//update_vec_subrange(U[i], c, Ha, Ha_start, Ha_end);
			//
			for ( long j=0 ; j<U[i].size(); j++)
#pragma omp atomic
				Ha[Ha_start+j] += c*U[i][j];
        }	
		delete[] b;
		delete[] cpvals;
		b = nullptr;
		cpvals = nullptr;
	}
	return Ha;
}




vec_t solve_delta(const vec_t& g, double* m, const mat_t& U, SparseMat* X, int r, 
				double lambda) {
	vec_t delta = vec_t(g.size(), 0.0);
	vec_t rr = copy_vec_t(g, -1.0);
	vec_t p = copy_vec_t(g);
//	vec_t p = copy_vec_t(rr, -1.0);
	double err = sqrt(norm(rr)) * 0.01;
//	cout << "break condition " << err << endl;
	double ttt = omp_get_wtime();
	for (int k = 1; k <= 10; ++k) {
		//vec_t Hp = copy_vec_t(p, lambda);
		double ttaa = omp_get_wtime();
		vec_t Hp = compute_Ha(p, m, U, X, r, lambda);

		double prod_p_Hp = dot(p, Hp);
		
		double alpha = -1.0 * dot(rr, p) / prod_p_Hp;
		
		delta = add_vec_vec(delta, p, 1.0, alpha);
		rr = add_vec_vec(rr, Hp, 1.0, alpha);
//		cout << "In CG, iteration " << k << " rr value:" << sqrt(norm(rr)) << endl;
		if (sqrt(norm(rr)) < err) {
			break;
		}
		double b = dot(rr, Hp) / prod_p_Hp;
		p = add_vec_vec(rr, p, -1.0, b);
	}
//	printf("AAA Time: %lf\n", omp_get_wtime()-ttt);
	return delta;
}

double* update_V(SparseMat* X, double lambda, double stepsize, int r, const mat_t& U,
				 mat_t& V, double& now_obj) {
	// update V while fixing U fixed
	double ttt = omp_get_wtime();
	double* m = comp_m(U, V, X, r);
//	printf("M time: %lf\n", omp_get_wtime()-ttt);
	double time = omp_get_wtime();
	
	//mat_t g = copy_mat_t(V, lambda);
	mat_t g = obtain_g(U, V, X, m, lambda);
//	printf("obtain_g_time %lf\n", omp_get_wtime()-time);
//	cout << "norm of g " << norm(g) << endl;
//	cout << "time for obtain_g function takes " << omp_get_wtime() - time << endl;
	
//	cout << "g is succesfully computed " << g.size() << "," << g[0].size() << endl;  
	// vectorize_mat function to convert g from mat_t into vec_t 
	vec_t g_vec;	
	vectorize_mat(g, g_vec);
//	cout << norm(g) - norm(g_vec) << endl;

 	//assert(norm(g) == norm(g_vec));	
	//cout << "vectorization is successful, now size is " << g_vec.size() << endl;
	// solve_delta function to implement conjugate gradient algorithm
	vec_t delta = solve_delta(g_vec, m, U, X, r, lambda);	
	// reshape function (not needed if implement mat_t substract vec_t function)	
//	cout << "solve_delta is okay" << endl;	
	//cout << "delta norm is " << norm(delta) << endl;
	double aatt = omp_get_wtime();
	double prev_obj = objective(m, U, V, X, lambda);	
	mat_t V_new;
//	cout << "stepsize is " << stepsize << endl;
//	cout << "norm of delta is " << norm(delta) << endl;
	// truncated newton till convergence
	for (int iter = 0; iter < 20; ++iter) {
		V_new = copy_mat_t(V, 1.0);
		mat_substract_vec(delta, stepsize, V_new);	
		delete[] m;
		m = comp_m(U, V_new, X, r);
		now_obj = objective(m, U, V_new, X, lambda);
	//	cout << "Line Search Iter " << iter << " Prev Obj " << prev_obj 
	//		 << " New Obj" << now_obj << " stepsize " << stepsize << endl;	
		if (now_obj < prev_obj) {
			V = copy_mat_t(V_new, 1.0);
			break;
		} else {
			stepsize /= 2.0;
		}
	}
	//printf("LINETIME: %lf\n", omp_get_wtime()-aatt);
//	printf("ALLALL time: %lf\n", omp_get_wtime()-ttt);
	return m;
}

double* obtain_g_u(long i, const mat_t& V, SparseMat* X, double* m, int r, double lambda, 
					double* D, vec_t& g, long& cc) {
	// cc is the number of pariwise comparisons for items of different ratings
	// should be upper bounded by num_pairs (or D.size())
	//cout << "entering obtain_g_u" << endl;
	
	double* vals = X->vals;
	long* rows = X->rows;
	long* index = X->index;
	long start = *(index + i);
	long end = *(index + i + 1) - 1;
	//cout << start << " " << end << endl;
	long len = end - start + 1;

	double val_j, val_k;
    double y_ijk, mask, s_jk;	
	double* t;

	t = new double[len]; 	// t is pointer to array length of len
	fill(t, t + len, 0.0);
	for (long j = start; j <= end - 1; ++j) {
		val_j = *(vals + j);
		for (long k = j + 1; k <= end; ++k) {
			val_k = *(vals + k);
			if (val_j == val_k) {
				continue;
         	} 
			/*else if (val_j > val_k) {
                y_ijk = 1.0;
            } else {
                y_ijk = -1.0;
            }*/
			mask = *(m + j) - *(m + k);
//            mask *= y_ijk;
			if ( val_k > val_j )
				mask = -mask;
			if (mask < 1.0) {
				D[cc] = 1.0;
//				s_jk = 2*(1-mask)*y_ijk;
				s_jk = 2*(1-mask);
				if ( val_k > val_j )
					s_jk = -s_jk;
				*(t + j - start) -= s_jk;
				*(t + k - start) += s_jk;
//				s_jk = 2.0 * (mask - 1);
//				*(t + j - start) += s_jk;
//				*(t + k - start) -= s_jk;
			}
			cc++;
		}
	}
	//cout << "first part finished" << endl;
	for (long k = 0; k < len; ++k) {
		long j = *(rows + start + k);
		double c = *(t + k); 
		// we want g += c * V[:,j];
		g = add_vec_vec(g, V[j], 1.0, c);
	}

	delete[] t;
	t = nullptr;
	return D;
}

double objective_u(long i, double* mm, const vec_t& ui, SparseMat* X, double lambda) {
	double res = 0.0;
	res += lambda / 2.0 * norm(ui);

	double* vals = X->vals;
	long* index = X->index;
	long start, end;
	double y_ijk, mask;
	start = *(index + i);
	end = *(index + i + 1) - 1;
	for (long j = start; j <= end - 1; ++j) {
		double val_j = *(vals + j);
		for (long k = j + 1; k <= end; ++k) {
			double val_k = *(vals + k);
			if (val_j == val_k) {
				continue;
			} /*else if (val_j > val_k) {
				y_ijk = 1.0;	
			} else {
				y_ijk = -1.0;
			}*/
			mask = *(mm + j - start) - *(mm + k - start);
			//mask *= y_ijk;
			if ( val_j < val_k )
				mask =-mask;
			if (mask < 1.0) {
				res += (1.0 - mask) * (1.0 - mask);
			}
		}
	}
	return res;
}

// unlike compute_Ha(), we don't need m, since no product of ui, vj will be used
vec_t obtain_Hs(long i, const vec_t& s, double* D, const mat_t& V, SparseMat* X, 
				double* m, int r, double lambda) {
	vec_t Hs = copy_vec_t(s, lambda);
	
	double* vals = X->vals;
	long* rows = X->rows;
	long* index = X->index;
	long start = *(index + i);
	long end = *(index + i + 1) - 1;
	long len = end - start + 1;
	double val_j, val_k;
	double ddd;
	double* b;
	double* cpvals;

	b = new double[len];  // to precompute ui*a
	for (long k = 0; k < len; ++k) {
		long j = *(rows + start + k);
		double res = 0.0;
		for (int k = 0; k < r; ++k) {
			res += s[k] * V[j][k];
		}
		*(b + k) = res;
	}

	cpvals = new double[len];
	fill(cpvals, cpvals + len, 0.0);

	long cc = 0;
	for (long j = start; j < end; ++j) {
		val_j = *(vals + j);
		for (long k = j + 1; k <= end; ++k) {
            val_k = *(vals + k);
			double y_ijk;
			if (val_j == val_k) {
				continue;
            } /*else if (val_j > val_k) {
				y_ijk = 1.0;
			} else {
				y_ijk = -1.0;
			}*/
			double mask = *(m + j) - *(m + k);
//			mask *= y_ijk;
			if ( val_j < val_k)
				mask = -mask;

			//if (mask < 1.0) {
            if (D[cc] > 0.0) {
            	ddd = *(b + j - start) - *(b + k - start);
            	ddd *= 2.0;
            	*(cpvals + j - start) += ddd;
                *(cpvals + k - start) -= ddd;
            }
			cc++;
        }
	}
	for (long k = 0; k < len; ++k) {
        long j = *(rows + start + k);
        double c = *(cpvals + k);
        Hs = add_vec_vec(Hs, V[j], 1.0, c);
	}	
	delete[] b;
	delete[] cpvals;
	b = nullptr;
	cpvals = nullptr;
	return Hs;
}

vec_t solve_delta_u(long i, vec_t& g, double* D, const mat_t& V, SparseMat* X, 
					double* m, int r, double lambda) {
	vec_t delta(g.size(), 0.0);
	vec_t rr = copy_vec_t(g, -1.0);
	vec_t p = copy_vec_t(rr, -1.0);
	double err = sqrt(norm(rr)) * 0.01;
//	cout << "break condition " << err << endl;
	for (int k = 1; k <= 10; ++k) {
		vec_t Hp = obtain_Hs(i, p, D, V, X, m, r, lambda);
		//vec_t Hp = compute_Ha(p, m, U, X, r, lambda);
		double prod_p_Hp = dot(p, Hp);
		double alpha = -1.0 * dot(rr, p) / prod_p_Hp;		
		delta = add_vec_vec(delta, p, 1.0, alpha);
		rr = add_vec_vec(rr, Hp, 1.0, alpha);
//		cout << "In CG, iteration " << k << " rr value:" << sqrt(norm(rr)) << endl;
		if (sqrt(norm(rr)) < err) {
			break;
		}
		double b = dot(rr, Hp) / prod_p_Hp;
		p = add_vec_vec(rr, p, -1.0, b);
	}
	return delta;
}


vec_t update_u(long i, const mat_t& V, SparseMat* X, double* m, int r, 
			  double lambda, double stepsize, const vec_t& ui, double& obj_u_new) {
	//cout << "enter update_u " << i << endl;
	long* index = X->index;
	long start = *(index + i);
	long end = *(index + i + 1) - 1;
	long len = end - start + 1;
//	cout << "User " << i << " has rated " << len << " items " << endl; 
	size_t num_pairs = static_cast<size_t>(len * (len - 1) / 2);
	//cout << "num_pairs " << num_pairs << endl;
	// use D to store mask results of pairwise comparison to save time
	// bad allocator error, could be too large for stack space
	// vec_t D = vec_t(num_pairs, 0.0);
	double* D = new double[num_pairs];
	fill(D, D + num_pairs, -1.0);
	//cout << "initializing D is okay" <<endl;
	// cc is the number of pariwise comparisons for items of different ratings
	// should be upper bounded by num_pairs (or D.size())
	long cc = 0;
	vec_t g = copy_vec_t(ui, lambda);
	D = obtain_g_u(i, V, X, m, r, lambda, D, g, cc);
//	cout << "norm of g for u0 is " << norm(g) << endl;
//	printf("g: ");
//	for ( int i=0 ; i<g.size() ; i++)
//		printf("%lf ", g[i]);
//	printf("\n");
	double* mm = compute_mm(i, ui, V, X, r);
	double prev_obj = objective_u(i, mm, ui, X, lambda);
//	cout << "prev obj is " << prev_obj << endl;
	if (cc == 0 || norm(g) < 0.0001) {
		obj_u_new = prev_obj;
		delete[] D;
		delete[] mm;
		mm = nullptr;
		D = nullptr;
		return ui;
	}
	vec_t delta = solve_delta_u(i, g, D, V, X, m, r, lambda);
	vec_t ui_new = copy_vec_t(ui, 1.0);
	for (int iter = 0; iter < 20; ++iter) {
		ui_new = add_vec_vec(ui, delta, 1.0, -stepsize);
		//update_m(i, ui_new, V, X, r, m);
	//	cout << "ui_new norm is " <<norm(ui_new) << endl;
		delete[] mm;
		mm = compute_mm(i, ui_new, V, X, r);
		
		obj_u_new = objective_u(i, mm, ui_new, X, lambda);
		
//		cout << "Line Search Iter " << iter << " Prev Obj " << prev_obj
//		             << " New Obj" << obj_u_new << " stepsize " << stepsize << endl;
		
		if (obj_u_new < prev_obj) {
			break;
		} else {
			stepsize /= 2.0;
		}
	}
	delete[] D;
	D = nullptr;
	mm = nullptr;
	return ui_new;

}

mat_t update_U(SparseMat* X, double* m, double lambda, double stepsize, int r, const mat_t& V, 
				const mat_t& U, double& now_obj) {
	// update U while fixing V
	//cout << "entering update_U" <<endl;
	double total_obj_new = 0.0;
	double obj_u_new = 0.0;
	long d1 = X->d1;
	mat_t U_new = copy_mat_t(U, 1.0);
//	for (long i = 0; i < 1; ++i) {
#pragma omp parallel for schedule(kind) reduction(+:total_obj_new)
	for (long i = 0; i < d1; ++i) {
		// modify U[i], obj_u_new inside update_u()
		vec_t ui_new = update_u(i, V, X, m, r, lambda, stepsize, U[i], obj_u_new);
		for (int k = 0; k < r; ++k) {
			U_new[i][k] = ui_new[k];
//			printf("k: %lf\n", ui_new[k]);
		}
		total_obj_new += obj_u_new;
	}

	total_obj_new += lambda / 2.0 * norm(V);
	now_obj = total_obj_new;
	return U_new;

}
	


// Primal-CR Algorithm
void pcr(smat_t& R, mat_t& U, mat_t& V, testset_t& T, parameter& param) {
	//cout<<"enter pcr"<<endl;
	int r = param.k;
	double lambda = param.lambda;
	double stepsize = param.stepsize;
	int ndcg_k = param.ndcg_k;
	double now_obj;
	double totaltime = 0.0;
	cout << "stepsize is " << stepsize << " and ndcg_k is " << ndcg_k << endl;
	// X: d1 by d2 sparse matrix, ratings
	// U: r by d1 dense
	// V: r by d2 dense
	// X ~ U^T * V

	omp_set_num_threads(param.threads);
	cout << "using " << omp_get_max_threads() << " threads. " << endl;

	SparseMat* X = convert(R);  // remember to free memory X in the end by delete X; set X = NULL;
	SparseMat* XT = convert(T, X->d1, X->d2);
	long nnz = (*X).nnz;
	cout << nnz << endl;
	double ttt = omp_get_wtime();
	double* m = comp_m(U, V, X, r);
//	printf("aaa time %lf\n", omp_get_wtime()-ttt);

//	printf("m[5]: %lf\n", m[4]);

	double time = omp_get_wtime();
	now_obj = objective(m, U, V, X, lambda);
//	printf("time for obj %lf\n", omp_get_wtime()-time);
	
	cout << "Iter 0, objective is " << now_obj << endl;

	double time1 = omp_get_wtime();
	pair<double, double> eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
	cout << "(Training) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
	printf("time for training error is %lf\n", omp_get_wtime()-time);

	eval_res = compute_pairwise_error_ndcg(U, V, XT, ndcg_k);
	cout << "(Testing) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;

	//int num_iter = 10;
	int num_iter = 20;
	
	/*
	mat_t g = copy_mat_t(V);
	
	cout << g.size() << ", g, " << g[0].size() << endl;
	for (long i = 0; i < X->d2; ++i) {
		cout << g[i][0] << endl;
	}
	*/
	
	double total_time = 0.0;
	
	for (int iter = 1; iter < num_iter; ++iter) {
		time = omp_get_wtime();
		// need to free space pointer m points to before pointing it to another memory
		delete[] m;
		m = update_V(X, lambda, stepsize, r, U, V, now_obj);
//		printf("update_V_time %lf\n", omp_get_wtime()-time);
		//cout << "Iter " << iter << " update_V " << "Time " << omp_get_wtime() - time << " Objective is " << now_obj << endl;
		//m = comp_m(U, V, X, r);
		U = update_U(X, m, lambda, stepsize, r, V, U, now_obj);
		//m = comp_m(U, V, X, r);
		//cout << (now_obj - objective(m, U, V, X, lambda)) << endl;
		//cout << "Iter " << iter << " update_U " << "Time " << omp_get_wtime() - time << " Obj " << now_obj << endl;
		total_time += omp_get_wtime() - time;
		cout << "Iter " << iter << ": Total Time " << total_time << " Obj " << now_obj << endl;
		eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
		cout << "(Training) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
		eval_res = compute_pairwise_error_ndcg(U, V, XT, ndcg_k);
		cout << "(Testing) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
	}
	
	delete[] m;
	delete X;
	//m = NULL;
	//X = NULL;
	m = nullptr;
	X = nullptr;
	return;
}
