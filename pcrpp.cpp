#include "util.h"
#include "pmf.h"

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
		mm[i] = *(m + i);
		perm_ind[cc] == cc;
		++cc;
	}
	assert(perm_ind.size() == len);
	sort(perm_ind.begin(), perm_ind.end(), 
		[&mm](long idx1, long idx2) {
			return mm[idx1] < mm[idx2];	
		}
	);
	sort(mm.begin(), mm.end());
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
	for (long i = 0; i < d1; ++i) {
		start = *(index + i);
        end = *(index + i + 1) - 1;
        len = end - start + 1;
		// use unordered_set to get rid of duplicate levels
		// levels are in increasing order
		vector<long> levels = find_levels(vals, start, end);		
		// find number of rating levels (ASSUMPTION: a small number)
		long num_levels = static_cast<long>(levels.size());
		// sort mm (m for i-th user) 
		vector<long> perm_ind(len, 0.0);
		vec_t mm_sorted = get_sorted_mm(m, start, end, len, perm_ind);
		// get corresponding vals, d2bar in the same order of mm
		// to be consistent order (useful for future calculation)
		vector<long> vals_sorted = get_sorted_vals(vals, start, len, perm_ind);	
		vector<long> d2bar_sorted = get_sorted_d2bar(rows, start, len, perm_ind);
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
			for (long k = 0; k < now_val - 1; ++k) {
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

double* update_V_new(SparseMat* X, double lambda, double stepsize, int r, 
			const mat_t& U, mat_t& V, double& now_obj) {
	double* m = comp_m_new(U, V, X, r);
	mat_t g = obtain_g_new(U, V, X, m, lambda);
	cout << "norm of g" << norm(g) << endl;
	return m;
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
    /*now_obj = objective(m, U, V, X, lambda);
	cout << "Iter 0, objective is " << now_obj << endl;
    pair<double, double> eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
    cout << "(Training) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
    eval_res = compute_pairwise_error_ndcg(U, V, XT, ndcg_k);
    cout << "(Testing) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
*/
    int num_iter = 2;

	double total_time = 0.0;

    for (int iter = 1; iter < num_iter; ++iter) {
        time = omp_get_wtime();
        // need to free space pointer m points to before pointing it to another memory
        delete[] m;
        m = update_V_new(X, lambda, stepsize, r, U, V, now_obj);
        /*U = update_U(X, m, lambda, stepsize, r, V, U, now_obj);
        
		eval_res = compute_pairwise_error_ndcg(U, V, X, ndcg_k);
        cout << "(Training) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
        eval_res = compute_pairwise_error_ndcg(U, V, XT, ndcg_k);
        cout << "(Testing) pairwise error is " << eval_res.first << " and ndcg is " << eval_res.second << endl;
    	*/
	}

    delete[] m;
    delete X;
    //m = NULL;
    //X = NULL;
    m = nullptr;
    X = nullptr;
    return;

}



