#include "util.h"
#include "pmf.h"


struct infor_ui {
	//vector<long> levels;
	long num_levels;
	//vector<long> perm_ind;
	vec_t mm_sorted;
	vector<long> vals_sorted;
	vector<long> d2bar_sorted;
	vector<long> count_right;
	long len;
};


double* comp_m_new(const mat_t& U, const mat_t& V, SparseMat* X, int r) {
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


vector<long> find_levels(double* vals, long start, long end) {
	unordered_set<long> levels;
	for (long i = start; i <= end; ++i) {
		levels.insert(lround(vals[i]));
	}
	vector<long> res;
	for (auto it = levels.begin(); it != levels.end(); ++it) {
		res.push_back(*it);
	}
	sort(res.begin(), res.end());
	return res;
}


vec_t get_sorted_mm(double* m, long start, long end, long len, 
			vector<long>& perm_ind) {
	vec_t mm(static_cast<size_t>(len), 0.0);
	long cc = 0;
	for (long i = start; i <= end; ++i) {
		mm[cc] = *(m + i);
		perm_ind[cc] = cc;
		++cc;
	}
	//cout << "size of perm_ind " << perm_ind.size()<<endl;
	assert(perm_ind.size() == mm.size());
/*	for (auto i : perm_ind) {
		cout << i << ",";
	}
*/
	sort(perm_ind.begin(), perm_ind.end(), 
		[&mm](size_t idx1, size_t idx2) {
			return mm[idx1] < mm[idx2];	
		}
	);
	/*cout << endl;
	for (long i = 0; i < len; ++i) {
		cout << *(m + start + i) << ",";
	}
	cout << endl;
	for (auto i : mm) {
		cout << i << ",";
	}*/
	sort(mm.begin(), mm.end());
	//cout << "returning from get_sorted_mm" <<endl;
	return mm;	
}


// get correponding item ratings according to sorted mm order
vector<long> get_sorted_vals(double* vals, long start, long len,
		 const vector<long>& perm_ind) {
	vector<long> vals_sorted(perm_ind.size(), 0);
	for (long i = 0; i < len; ++i) {
		long j = perm_ind[i];
		vals_sorted[i] = lround(*(vals + start + j));
	}
	return vals_sorted;
} 


// get correponding item index according to sorted mm order
vector<long> get_sorted_d2bar(long* rows, long start, 
                     long len, const vector<long>& perm_ind) {
	vector<long> d2bar_sorted(perm_ind.size(), 0);
	for (long i = 0; i < len; ++i) {
		long j = perm_ind[i];
		d2bar_sorted[i] = *(rows + start + j);
	}
	return d2bar_sorted;
}

vec_t get_sorted_b(const vec_t& b, long start, long len, 
			const vector<long>& perm_ind) {
	vec_t b_sorted(perm_ind.size(), 0);
	for (long i = 0; i < len; ++i) {
		long j = perm_ind[i];
		b_sorted[i] = b[j]; 
	}
	return b_sorted;
}

vec_t get_levels_sum(const vec_t& mm_sorted, const vector<long> vals_sorted,
			long num_levels, long len) {
	vec_t levels_sum(static_cast<size_t>(num_levels), 0.0);
	for (long j = 0; j < len; ++j) {
		long level_j = vals_sorted[j];
		levels_sum[level_j] += mm_sorted[j];
	}
	return levels_sum;
}

vector<long> get_count_right(const vector<long> vals_sorted, 
				long num_levels, long len) {
	vector<long> count_right(static_cast<size_t>(num_levels), 0);
	for (long j = 0; j < len; ++j) {
		long level_j = vals_sorted[j];
		count_right[level_j] += 1;
	}
	return count_right;
}


mat_t obtain_g_new(const mat_t& U, const mat_t& V, SparseMat* X, 
			double* m, double lambda) {
	mat_t g = copy_mat_t(V, lambda);
	long d1 = X->d1;
    double* vals = X->vals;
    long* index = X->index;
    long* rows = X->rows;
	long start, end, len;
	long num_levels;
	
	for (long i = 0; i < d1; ++i) {
//		cout << "user id " << i << endl;
		start = *(index + i);	
		end = *(index + i + 1) - 1;
		len = end - start + 1;
//		cout << start << "," << end << "," << len << endl;
		// use unordered_set to get rid of duplicate levels
		// levels are in increasing order
		vector<long> levels = find_levels(vals, start, end);		
		// find number of rating levels (ASSUMPTION: a small number)
		num_levels = static_cast<long>(levels.size());
//		cout << "number of levels " << num_levels << endl;

		// sort mm (m for i-th user) 
		vector<long> perm_ind(len, 0);
		vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
	//	cout << "mm is sorted, last element is " << mm_sorted[len - 1] << " perm_ind last element is " 
	//		<< perm_ind[len - 1] << endl;
		
		// get corresponding vals, d2bar in the same order of mm
		// to be consistent order (useful for future calculation)
		vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);	

	//	cout << "vals is sorted, last element is " << vals_sorted[len - 1] << endl;
		vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
	//	cout << "d2bar is sorted, last element is " << d2bar_sorted[len - 1] << endl;
/*		for (auto j : d2bar_sorted) {
			cout << j << ",";
		}	
*/			
		// this to set item ratings to be continuous integers
		// transformation from original ratings to new ones (preserving order)
		for (long j = 0; j < len; ++j) {
			for (long k = 0; k < num_levels; ++k) {
				if (vals_sorted[j] == levels[k]) {
					vals_sorted[j] = k;
					break;
				}
			}
		}
		// obtain levels_sum containing sum of all the mm values at each rating level
		vec_t levels_sum = get_levels_sum(mm_sorted, vals_sorted, num_levels, len);
		vec_t now_right_sum(levels_sum);

		vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);

		// now_left pointer is to find mm_sorted[now_left] < now_cut + 1.0
		long now_left = 0;
		
		// now_right pointer is to find mm_sorted[now_right] < now_cut - 1.0
		long now_right = 0;

		// record the count of ratings same as the value pointer points to 
		// (one pointer points to three array: mm_sorted, vals_sorted, d2bar_sorted)
		// the initial pointer is at mm_sorted.begin()
		// count_left refers to left to the pointer (including boundary)
		vector<long> count_left(static_cast<size_t>(num_levels), 0);
		
		// count_right refers to count to the right
		vector<long> count_right = get_count_right(vals_sorted, num_levels, len);
/*		cout << endl;
		for (auto j : count_right) {
			cout << j << ",";
		} 	*/
		for (long j = 0; j < len; ++j) {
			double now_cut = mm_sorted[j];
			long now_val = vals_sorted[j];
			long level;
			while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
				level = vals_sorted[now_left];
				now_left_sum[level] += mm_sorted[now_left];
				count_left[level] += 1;
				now_left += 1;
			}
			while (now_right < len && mm_sorted[now_right] < now_cut - 1.0) {
				level = vals_sorted[now_right];
				now_right_sum[level] -= mm_sorted[now_right];
				count_right[level] -= 1;
				now_right += 1;
			}
			double c = 0.0;
			for (long k = 0; k <= now_val - 1; ++k) {
				c += (count_right[k] * (mm_sorted[j] - 1.0) - now_right_sum[k]);
			}
			for (long k = now_val + 1; k < num_levels; ++k) {
				c += (count_left[k] * (mm_sorted[j] + 1.0) - now_left_sum[k]);
			}
			long p = d2bar_sorted[j];
			c *= 2.0;
			update_mat_add_vec(U[i], c, p, g);

		}
	
	}	
	return g;
}


vec_t compute_Ha_new(const vec_t& a, double* m, const mat_t& U, SparseMat* X,
              int r, double lambda) {
	vec_t Ha = copy_vec_t(a, lambda);
	long d1 = X->d1;
	double* vals = X->vals;
	long* rows = X->rows;
	long* index = X->index;
	long start, end, len;
	long a_start, a_end;
	long Ha_start, Ha_end;
	long num_levels;
	for (long i = 0; i < d1; ++i) {
		start = *(index + i);
		end = *(index + i + 1) - 1;
		len = end - start + 1;
		vec_t b(static_cast<size_t>(len), 0.0);
		long cc = 0;
		for (long k = 0; k < len; ++k) {
			long q = *(rows + start + k);
			a_start = q * r;
			a_end = (q + 1) * r - 1;
			b[cc++] = vec_prod_array(U[i], a, a_start, a_end);
		}
		vector<long> levels = find_levels(vals, start, end);
		num_levels = static_cast<long>(levels.size());
		vector<long> perm_ind(len, 0);
		vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
		vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);
		vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
		vec_t b_sorted = get_sorted_b(b, start, len, perm_ind);
		for (long j = 0; j < len; ++j) {
			for (long k = 0; k < num_levels; ++k) {
				if (vals_sorted[j] == levels[k]) {
					vals_sorted[j] = k;
					break;
				}
			}
		}
		vec_t levels_sum = get_levels_sum(b_sorted, vals_sorted, num_levels, len);
		vec_t now_right_sum(levels_sum);
		vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);
		long now_left = 0;
		long now_right = 0;
		vector<long> count_left(static_cast<size_t>(num_levels), 0);
		vector<long> count_right = get_count_right(vals_sorted, num_levels, len);
		for (long j = 0; j < len; ++j) {
			double now_cut = mm_sorted[j];
			long now_val = vals_sorted[j];
			long level;
			while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
				level = vals_sorted[now_left];
				now_left_sum[level] += b_sorted[now_left];
				count_left[level] += 1;
				now_left += 1;
			}
			while (now_right < len && mm_sorted[now_right] < now_cut - 1.0) {
				level = vals_sorted[now_right];
				now_right_sum[level] -= b_sorted[now_right];
				count_right[level] -= 1;
				now_right += 1;
			}
			double c = 0.0;
			for (long k = 0; k <= now_val - 1; ++k) {
				c += (count_right[k] * b_sorted[j] - now_right_sum[k]);
			}
			for (long k = now_val + 1; k < num_levels; ++k) {
				c += (count_left[k] * b_sorted[j] - now_left_sum[k]);
			}
			long p = d2bar_sorted[j];
			c *= 2.0;
			Ha_start = p * r;
			Ha_end = (p + 1) * r - 1;
			update_vec_subrange(U[i], c, Ha, Ha_start, Ha_end);
		}
	}
    return Ha;
}


vec_t solve_delta_new(const vec_t& g, double* m, const mat_t& U, 
		SparseMat* X, int r, double lambda) {
    vec_t delta = vec_t(g.size(), 0.0);
    vec_t rr = copy_vec_t(g, -1.0);
    vec_t p = copy_vec_t(g);
    double err = sqrt(norm(rr)) * 0.01;
  //  cout << "break condition " << err << endl;
	double ttt = omp_get_wtime();
	for (int k = 1; k <= 10; ++k) {
        vec_t Hp = compute_Ha_new(p, m, U, X, r, lambda);
        double prod_p_Hp = dot(p, Hp);
        double alpha = -1.0 * dot(rr, p) / prod_p_Hp;
        delta = add_vec_vec(delta, p, 1.0, alpha);
        rr = add_vec_vec(rr, Hp, 1.0, alpha);
    //	cout << "In CG, iteration " << k << " rr value:" << sqrt(norm(rr)) << endl;
        if (sqrt(norm(rr)) < err) {
            break;
        }
        double b = dot(rr, Hp) / prod_p_Hp;
        p = add_vec_vec(rr, p, -1.0, b);
    }
    //printf("AAA Time: %lf\n", omp_get_wtime()-ttt);
    return delta;
}


double objective_new(double* m, const mat_t& U, const mat_t& V,
        SparseMat* X, double lambda) {
    double res = 0.0;
    double norm_U = norm(U);
    double norm_V = norm(V);
    res += lambda * (norm_U + norm_V) / 2.0;
    long d1 = X->d1;
    double* vals = X->vals;
    long* index = X->index;
    long start, end, len;
    long num_levels;
    for (long i = 0; i < d1; ++i) {
        start = *(index + i);
        end = *(index + i + 1) - 1;
        len = end - start + 1;
        vector<long> levels = find_levels(vals, start, end);
        num_levels = static_cast<long>(levels.size());
        vector<long> perm_ind(len, 0);
        vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
        vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);
        for (long j = 0; j < len; ++j) {
            for (long k = 0; k < num_levels; ++k) {
                if (vals_sorted[j] == levels[k]) {
                    vals_sorted[j] = k;
                    break;
                }
            }
        }
        long now_left = 0;
        vector<long> count_left(static_cast<size_t>(num_levels), 0);
        vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);
        vec_t now_left_sqsum(static_cast<size_t>(num_levels), 0.0);
        for (long j = 0; j < len; ++j) {
            double now_cut = mm_sorted[j];
            long now_val = vals_sorted[j];
            long level;
            while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
                level = vals_sorted[now_left];
                now_left_sum[level] += (mm_sorted[now_left] - 1.0);
                now_left_sqsum[level] += pow(mm_sorted[now_left] - 1.0, 2.0);
                count_left[level] += 1;
                now_left += 1;
            }
            for (long k = now_val + 1; k < num_levels; ++k) {
                res += (count_left[k] * pow(now_cut, 2.0) - 2.0 *
                    now_cut * now_left_sum[k] + now_left_sqsum[k]);
            }
        }
    }
    return res;
}


double* update_V_new(SparseMat* X, double lambda, double stepsize, int r, 
			const mat_t& U, mat_t& V, double& now_obj) {
	double* m = comp_m_new(U, V, X, r);
	mat_t g = obtain_g_new(U, V, X, m, lambda);
	//cout << "norm of g " << norm(g) << endl;
	vec_t g_vec;
	vectorize_mat(g, g_vec);
	vec_t delta = solve_delta_new(g_vec, m, U, X, r, lambda);
	//cout << "delta norm is " << norm(delta) << endl;
	double aatt = omp_get_wtime();
	double prev_obj = objective_new(m, U, V, X, lambda);
	mat_t V_new;
	for (int iter = 0; iter < 20; ++iter) {
        V_new = copy_mat_t(V, 1.0);
        mat_substract_vec(delta, stepsize, V_new);
        delete[] m;
        m = comp_m_new(U, V_new, X, r);
        now_obj = objective_new(m, U, V_new, X, lambda);
		//cout << "Line Search Iter " << iter << " Prev Obj " << prev_obj 
     	//	<< " New Obj" << now_obj << " stepsize " << stepsize << endl;  
        if (now_obj < prev_obj) {
            V = copy_mat_t(V_new, 1.0);
            break;
        } else {
            stepsize /= 2.0;
        }
    }
	//printf("LINETIME: %lf\n", omp_get_wtime()-aatt);
	return m;
}


infor_ui* precompute_ui(long i, const mat_t& V, SparseMat* X, double* m) {
    double* vals = X->vals;
    long* index = X->index;
    long* rows = X->rows;
    long start = *(index + i);
    long end = *(index + i + 1) - 1;
    long len = end - start + 1;
    vector<long> levels = find_levels(vals, start, end);
    long num_levels = static_cast<long>(levels.size());
    vector<long> perm_ind(len, 0);
    vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
    vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);
    vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
    for (long j = 0; j < len; ++j) {
        for (long k = 0; k < num_levels; ++k) {
            if (vals_sorted[j] == levels[k]) {
                vals_sorted[j] = k;
                break;
            }
        }
    }
    vector<long> count_right = get_count_right(vals_sorted, num_levels, len);
    infor_ui* p = new infor_ui();   // on heap is faster than on stack
    p->num_levels = num_levels;
    p->mm_sorted = mm_sorted;
    p->vals_sorted = vals_sorted;
    p->d2bar_sorted = d2bar_sorted;
    p->count_right = count_right;
    p->len = len;
    return p;
}


/*
struct infor_ui {
   // levels not needed later
   vector<long> levels;
    long num_levels;
    vector<long> perm_id;
    vec_t mm_sorted;
    vector<long> vals_sorted;
    vector<long> d2bar_sorted;
    vector<long> count_right;
}
*/

vec_t obtain_g_u_new(long i, const mat_t& V, double lambda,
			const vec_t& ui, infor_ui* p_infor_ui) {	
	vec_t g = copy_vec_t(ui, lambda);
	if (p_infor_ui->len == 0) {
		return vec_t(ui.size(), 0.0);
	}
	long num_levels = p_infor_ui->num_levels;
    vec_t mm_sorted = p_infor_ui->mm_sorted;
    vector<long> vals_sorted = p_infor_ui->vals_sorted;
    vector<long> d2bar_sorted = p_infor_ui->d2bar_sorted;
    vector<long> count_left(static_cast<size_t>(num_levels), 0);
	vector<long> count_right = p_infor_ui->count_right;
	long len = p_infor_ui->len;
	vec_t levels_sum = get_levels_sum(mm_sorted, vals_sorted, num_levels, len);
	vec_t now_right_sum(levels_sum);
	vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);
	long now_left = 0;
	long now_right = 0;
	for (long j = 0; j < len; ++j) {
		double now_cut = mm_sorted[j];
		long now_val = vals_sorted[j];
		long level;
		while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
			level = vals_sorted[now_left];
			now_left_sum[level] += mm_sorted[now_left];
			count_left[level] += 1;
			now_left += 1;
		}
		while (now_right < len && mm_sorted[now_right] < now_cut - 1.0) {
			level = vals_sorted[now_right];
			now_right_sum[level] -= mm_sorted[now_right];
			count_right[level] -= 1;
			now_right += 1;
		}
		double c = 0.0;
		for (long k = 0; k <= now_val - 1; ++k) {
			c += (count_right[k] * (mm_sorted[j] - 1.0) - now_right_sum[k]);
		}
		for (long k = now_val + 1; k < num_levels; ++k) {
			c += (count_left[k] * (mm_sorted[j] + 1.0) - now_left_sum[k]);
		}
		long p = d2bar_sorted[j];
		c *= 2.0;
		g = add_vec_vec(g, V[p], 1.0, c);
	}	
	return g;
}


double objective_u_new(long i, infor_ui* p_infor_ui, 
		const vec_t& ui, double lambda) {
	double res = 0.0;
	res += lambda / 2.0 * norm(ui);
	
	long num_levels = p_infor_ui->num_levels;
    vec_t mm_sorted = p_infor_ui->mm_sorted;
    vector<long> vals_sorted = p_infor_ui->vals_sorted;
    long len = p_infor_ui->len;
	
	long now_left = 0;
	vector<long> count_left(static_cast<size_t>(num_levels), 0);
	vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);
	vec_t now_left_sqsum(static_cast<size_t>(num_levels), 0.0);
	for (long j = 0; j < len; ++j) {
		double now_cut = mm_sorted[j];
		long now_val = vals_sorted[j];
		long level;
		while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
			level = vals_sorted[now_left];
			now_left_sum[level] += (mm_sorted[now_left] - 1.0);
			now_left_sqsum[level] += pow(mm_sorted[now_left] - 1.0, 2.0);
			count_left[level] += 1;
			now_left += 1;
		}
		for (long k = now_val + 1; k < num_levels; ++k) {
			res += (count_left[k] * pow(now_cut, 2.0) - 2.0 *
			now_cut * now_left_sum[k] + now_left_sqsum[k]);
		}
	}
	return res;
}


vec_t obtain_Hs_new(long i, const vec_t& s, const mat_t& V, 
		infor_ui* p_infor_ui, double lambda) {
	vec_t Hs = copy_vec_t(s, lambda);
	
	long num_levels = p_infor_ui->num_levels;
    vec_t mm_sorted = p_infor_ui->mm_sorted;
    vector<long> vals_sorted = p_infor_ui->vals_sorted;
    vector<long> d2bar_sorted = p_infor_ui->d2bar_sorted;
    vector<long> count_left(static_cast<size_t>(num_levels), 0);
    vector<long> count_right = p_infor_ui->count_right;
    long len = p_infor_ui->len;
    vec_t now_left_sum(static_cast<size_t>(num_levels), 0.0);
    long now_left = 0;
    long now_right = 0;
	
	vec_t b_sorted(static_cast<size_t>(len), 0.0);
	for (long k = 0; k < len; ++k) {
		b_sorted[k] = dot(s, V[d2bar_sorted[k]]);
	}
	vec_t levels_sum = get_levels_sum(b_sorted, vals_sorted, num_levels, len);
    vec_t now_right_sum(levels_sum);
	for (long j = 0; j < len; ++j) {
		double now_cut = mm_sorted[j];
		long now_val = vals_sorted[j];
		long level;
		while (now_left < len && mm_sorted[now_left] <= now_cut + 1.0) {
			level = vals_sorted[now_left];
			now_left_sum[level] += b_sorted[now_left];
			count_left[level] += 1;
			now_left += 1;
		}
		while (now_right < len && mm_sorted[now_right] < now_cut - 1.0) {
			level = vals_sorted[now_right];
			now_right_sum[level] -= b_sorted[now_right];
			count_right[level] -= 1;
			now_right += 1;
		}
		double c = 0.0;
		for (long k = 0; k <= now_val - 1; ++k) {
			c += (count_right[k] * b_sorted[j] - now_right_sum[k]);
		}
		for (long k = now_val + 1; k < num_levels; ++k) {
			c += (count_left[k] * b_sorted[j] - now_left_sum[k]);
		}
		long p = d2bar_sorted[j];
		c *= 2.0;
		Hs = add_vec_vec(Hs, V[p], 1.0, c); 
	}
	return Hs;
}


vec_t solve_delta_u_new(long i, vec_t& g, const mat_t& V,
		infor_ui* p_infor_ui, double lambda) {
    vec_t delta(g.size(), 0.0);
    vec_t rr = copy_vec_t(g, -1.0);
    vec_t p = copy_vec_t(rr, -1.0);
    double err = sqrt(norm(rr)) * 0.01;
    for (int k = 1; k <= 10; ++k) {
        vec_t Hp = obtain_Hs_new(i, p, V, p_infor_ui, lambda);
        double prod_p_Hp = dot(p, Hp);
        double alpha = -1.0 * dot(rr, p) / prod_p_Hp;
        delta = add_vec_vec(delta, p, 1.0, alpha);
        rr = add_vec_vec(rr, Hp, 1.0, alpha);
        if (sqrt(norm(rr)) < err) {
            break;
        }
        double b = dot(rr, Hp) / prod_p_Hp;
        p = add_vec_vec(rr, p, -1.0, b);
    }
    return delta;
}

/*
infor_ui* precompute_ui(long i, const mat_t& V, SparseMat* X, double* m) {
    double* vals = X->vals;
    long* index = X->index;
    long* rows = X->rows;
    long start = *(index + i);
    long end = *(index + i + 1) - 1;
    long len = end - start + 1;
    vector<long> levels = find_levels(vals, start, end);
    long num_levels = static_cast<long>(levels.size());
    vector<long> perm_ind(len, 0);
    vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
    vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);
    vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
    for (long j = 0; j < len; ++j) {
        for (long k = 0; k < num_levels; ++k) {
            if (vals_sorted[j] == levels[k]) {
                vals_sorted[j] = k;
                break;
            }
        }
    }
    vector<long> count_right = get_count_right(vals_sorted, num_levels, len);
    infor_ui* p = new infor_ui();   // on heap is faster than on stack
    p->num_levels = num_levels;
    p->mm_sorted = mm_sorted;
    p->vals_sorted = vals_sorted;
    p->d2bar_sorted = d2bar_sorted;
    p->count_right = count_right;
    p->len = len;
    return p;
}
*/


infor_ui* update_infor_ui(long i, const vec_t& ui_new, const mat_t& V,
			SparseMat* X, int r, double* mm){
	double* vals = X->vals;
    long* index = X->index;
    long* rows = X->rows;
    long start = *(index + i);
    long end = *(index + i + 1) - 1;
    long len = end - start + 1;
	
/*	
	double* mm = new double[len];
    for (long j = start; j <= end; ++j) { // should be "<=",
			//	bug resulting from copying code and inconsistency of end definition
			// in update_m, compute_mm in pcr.cpp
        long item_id = *(rows + j);
        double res = 0.0;
        for (int k = 0; k < r; ++k) {
            res += ui_new[k] * V[item_id][k];
        }
        *(mm + j - start) = res;
    }
*/	vector<long> perm_ind(len, 0);
	vec_t mm_sorted = get_sorted_mm(mm, 0, len - 1, len, perm_ind);
	vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);
	vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
	vector<long> levels = find_levels(vals, start, end);
    long num_levels = static_cast<long>(levels.size());
	for (long j = 0; j < len; ++j) {
        for (long k = 0; k < num_levels; ++k) {
            if (vals_sorted[j] == levels[k]) {
                vals_sorted[j] = k;
                break;
            }
        }
    }
    infor_ui* p = new infor_ui();   // on heap is faster than on stack
    p->num_levels = num_levels;
    p->mm_sorted = mm_sorted;
    p->vals_sorted = vals_sorted;
    p->d2bar_sorted = d2bar_sorted;
    p->len = len;		
	return p;
}

double* compute_mm_old(long i, const vec_t& ui_new, const mat_t& V, SparseMat* X, int r) {
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

double objective_u_old(long i, double* mm, const vec_t& ui, SparseMat* X, double lambda) {
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

vec_t update_u_new(long i, const mat_t& V, SparseMat* X, double* m, int r,
  double lambda, double stepsize, const vec_t& ui, double& obj_u_new) {
	infor_ui* p_infor_ui = precompute_ui(i, V, X, m);
//	cout << p_infor_ui->len << endl;
	vec_t g = obtain_g_u_new(i, V, lambda, ui, p_infor_ui);	
	//cout << "norm of g for u0 is " << norm(g) << endl;
	double prev_obj = objective_u_new(i, p_infor_ui, ui, lambda);
	//cout << "prev obj is " << prev_obj << endl;
	if (norm(g) < 0.0001) {
        obj_u_new = prev_obj;
        return ui;
    }
	vec_t delta = solve_delta_u_new(i, g, V, p_infor_ui, lambda);
	vec_t ui_new = copy_vec_t(ui, 1.0);
	double* mm;
    for (int iter = 0; iter < 20; ++iter) {
        ui_new = add_vec_vec(ui, delta, 1.0, -stepsize);
	//	cout << "ui_new norm is " <<norm(ui_new) << endl;
		
		mm = compute_mm_old(i, ui_new, V, X, r);
		
		infor_ui* p_infor_ui_new = update_infor_ui(i, ui_new, V, X, r, mm);
        //cout << "value of i " << i <<endl;
		obj_u_new = objective_u_new(i, p_infor_ui_new, ui_new, lambda);
        //mm = compute_mm_old(i, ui_new, V, X, r);
		//obj_u_new = objective_u_old(i, mm, ui_new, X, lambda);
		delete[] mm;
	//	cout << "Line Search Iter " << iter << " Prev Obj " << prev_obj
	//		<< " New Obj" << obj_u_new << " stepsize " << stepsize << endl;
		if (obj_u_new < prev_obj) {
            break;
        } else {
            stepsize /= 2.0;
        }
    }
	return ui;	
}


mat_t update_U_new(SparseMat* X, double* m, double lambda, 
			double stepsize, int r, const mat_t& V,
                const mat_t& U, double& now_obj) {
    double total_obj_new = 0.0;
    total_obj_new += lambda / 2.0 * norm(V);
    double obj_u_new = 0.0;
    long d1 = X->d1;
    mat_t U_new = copy_mat_t(U, 1.0);
    for (long i = 0; i < d1; ++i) {
        // modify U[i], obj_u_new inside update_u()
        vec_t ui_new = update_u_new(i, V, X, m, r, lambda, stepsize, U[i], obj_u_new);
        for (int k = 0; k < r; ++k) {
            U_new[i][k] = ui_new[k];
        }
        total_obj_new += obj_u_new;
    }
    now_obj = total_obj_new;
    return U_new;
}


void pcrpp(smat_t& R, mat_t& U, mat_t& V, testset_t& T, parameter& param) {
    //cout<<"enter pcr"<<endl;
    int r = param.k;
    double lambda = param.lambda;
    double stepsize = param.stepsize;
    int ndcg_k = param.ndcg_k;
    double now_obj = 0.0;
    double totaltime = 0.0;
    cout << "stepsize is " << stepsize << " and ndcg_k is " << ndcg_k << endl;	
    SparseMat* X = convert(R);
    SparseMat* XT = convert(T, X->d1, X->d2);
    long nnz = X->nnz;

    double* m = comp_m_new(U, V, X, r);
    double time = omp_get_wtime();
    now_obj = objective_new(m, U, V, X, lambda);
	cout << "Iter 0, objective is " << now_obj << endl;
    pair<double, double> eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
    cout << "(Training) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
    eval_res = compute_pairwise_error_ndcg(U, V, XT, ndcg_k);
    cout << "(Testing) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
    int num_iter = 20;

    double total_time = 0.0;

    for (int iter = 1; iter < num_iter; ++iter) {
        time = omp_get_wtime();
        // need to free space pointer m points to before pointing it to another memory
        delete[] m;
        m = update_V_new(X, lambda, stepsize, r, U, V, now_obj);
        U = update_U_new(X, m, lambda, stepsize, r, V, U, now_obj);
		eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
        total_time += omp_get_wtime() - time;
        cout << "Iter " << iter << ": Total Time " << total_time << " Obj " << now_obj << endl;
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



