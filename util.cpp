#include "util.h"
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))


// load utility for CCS RCS
void load(const char* srcdir, smat_t &R, testset_t &T, bool with_weights){
	// add testing later
	char filename[1024], buf[1024];
	sprintf(filename,"%s/meta",srcdir);
	FILE *fp = fopen(filename,"r");
	long m, n, nnz;
	fscanf(fp, "%ld %ld", &m, &n);

	fscanf(fp, "%ld %s", &nnz, buf);
	sprintf(filename,"%s/%s", srcdir, buf);
	R.load(m, n, nnz, filename, with_weights);

	if(fscanf(fp, "%ld %s", &nnz, buf)!= EOF){
		sprintf(filename,"%s/%s", srcdir, buf);
		T.load(m, n, nnz, filename);
	}
	fclose(fp);
	//double bias = R.get_global_mean(); R.remove_bias(bias); T.remove_bias(bias);
	return ;
}

// Save a mat_t A to a file in row_major order.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(mat_t A, FILE *fp, bool row_major){
	if (fp == NULL) 
		fprintf(stderr, "output stream is not valid.\n");
	long m = row_major? A.size(): A[0].size();
	long n = row_major? A[0].size(): A.size();
	fwrite(&m, sizeof(long), 1, fp);
	fwrite(&n, sizeof(long), 1, fp);
	vec_t buf(m*n);

	if (row_major) {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i) 
			for(size_t j = 0; j < n; ++j)
				buf[idx++] = A[i][j];
	} else {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i) 
			for(size_t j = 0; j < n; ++j)
				buf[idx++] = A[j][i];
	}
	fwrite(&buf[0], sizeof(double), m*n, fp);
}

// Load a matrix from a file and return a mat_t matrix 
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returened A  is stored in col_major order.
mat_t load_mat_t(FILE *fp, bool row_major){
	if (fp == NULL) 
		fprintf(stderr, "input stream is not valid.\n");
	long m, n; 
	fread(&m, sizeof(long), 1, fp);
	fread(&n, sizeof(long), 1, fp);
	vec_t buf(m*n);
	fread(&buf[0], sizeof(double), m*n, fp);
	mat_t A;
	if (row_major) {
		A = mat_t(m, vec_t(n));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i) 
			for(size_t j = 0; j < n; ++j)
				A[i][j] = buf[idx++];
	} else {
		A = mat_t(n, vec_t(m));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i) 
			for(size_t j = 0; j < n; ++j)
				A[j][i] = buf[idx++];
	}
	return A;
}
void initial(mat_t &X, long n, long k){
	X = mat_t(n, vec_t(k));
	srand48(0L);
	for(long i = 0; i < n; ++i){
		for(long j = 0; j < k; ++j)
			X[i][j] = 0.1*drand48(); //-1;
			//X[i][j] = 0; //-1;
	}
}

void initial_col(mat_t &X, long k, long n){
	X = mat_t(k, vec_t(n));
	srand48(0L);
	for(long i = 0; i < n; ++i)
		for(long j = 0; j < k; ++j)
			X[j][i] = 0.1*drand48();
}

double dot(const vec_t &a, const vec_t &b){
	double ret = 0;
#pragma omp parallel for 
	for(int i = a.size()-1; i >=0; --i)
		ret+=a[i]*b[i];
	return ret;
}
double dot(const mat_t &W, const int i, const mat_t &H, const int j){
	int k = W.size();
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*H[t][j];
	return ret;
}
double dot(const mat_t &W, const int i, const vec_t &H_j){
	int k = H_j.size();
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*H_j[t];
	return ret;
}
double norm(const vec_t &a){
	double ret = 0;
	for(int i = a.size()-1; i >=0; --i)
		ret+=a[i]*a[i];
	return ret;
}
double norm(const mat_t &M) {
	double reg = 0;
	for(int i = M.size()-1; i>=0; --i) reg += norm(M[i]);
	return reg;
}
double calloss(const smat_t &R, const mat_t &W, const mat_t &H){
	double loss = 0;
	int k = H.size();
	for(long c = 0; c < R.cols; ++c){
		for(long idx = R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
			double diff = - R.val[idx];
			diff += dot(W[R.row_idx[idx]], H[c]);
			loss += diff*diff;
		}
	}
	return loss;
}
double calobj(const smat_t &R, const mat_t &W, const mat_t &H, const double lambda, bool iscol){
	double loss = 0;
	int k = iscol?H.size():0;
	vec_t Hc(k);
	for(long c = 0; c < R.cols; ++c){
		if(iscol) 
			for(int t=0;t<k;++t) Hc[t] = H[t][c];
		for(long idx = R.col_ptr[c]; idx < R.col_ptr[c+1]; ++idx){
			double diff = - R.val[idx];
			if(iscol)
				diff += dot(W, R.row_idx[idx], Hc);
			else 
				diff += dot(W[R.row_idx[idx]], H[c]);
			loss += (R.with_weights? R.weight[idx] : 1.0) * diff*diff;
		}
	}
	double reg = 0;
	if(iscol) {
		for(int t=0;t<k;++t) {
			for(long r=0;r<R.rows;++r) reg += W[t][r]*W[t][r]*R.nnz_of_row(r);
			for(long c=0;c<R.cols;++c) reg += H[t][c]*H[t][c]*R.nnz_of_col(c);
		}
	} else {
		for(long r=0;r<R.rows;++r) reg += R.nnz_of_row(r)*norm(W[r]);
		for(long c=0;c<R.cols;++c) reg += R.nnz_of_col(c)*norm(H[c]);
	}
	reg *= lambda;
	return loss + reg;
}

double calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
	for(size_t idx = 0; idx < nnz; ++idx){
		err = -testset[idx].v;
		if(iscol)
			err += dot(W, testset[idx].i, H, testset[idx].j);
		else 
			err += dot(W[testset[idx].i], H[testset[idx].j]);
		rmse += err*err;
	}
	return sqrt(rmse/nnz);
}

double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse/nnz);
}

double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for reduction(+:rmse)
	for(size_t idx = 0; idx < nnz; ++idx){
		testset[idx].v -= Wt[testset[idx].i]*Ht[testset[idx].j] - oldWt[testset[idx].i]*oldHt[testset[idx].j];
		rmse += testset[idx].v*testset[idx].v;
	}
	return sqrt(rmse/nnz);
}


// In d1 by d2 matrix R, where d1 is number of users and d2 is number of movies 
SparseMat* convert(smat_t &R) {
    long d1 = R.rows;
    long d2 = R.cols;
    long nnz = R.nnz;
	cout << "right before X is created" << endl;
    SparseMat *X = new SparseMat(d1, d2, nnz);
    cout << "X is created: users " << (*X).d1 << ",items " << (*X).d2 << ",nnz "<<(*X).nnz << endl;
	// transpose to get the same format as X
    R = R.transpose();
	long cc = 0;
    for (long j = 0; j < d1; ++j){
        (*X).index[j] = cc;
        for (long idx = R.col_ptr[j]; idx < R.col_ptr[j + 1]; ++idx){
            (*X).vals[cc] = R.val[idx];
            (*X).cols[cc] = j;
            (*X).rows[cc] = static_cast<long>(R.row_idx[idx]);
            cc++;
        }
    }
    if (cc == nnz){
        cout << "cc is correct in convert function" << endl;
    } else{
        cout << "something is wrong with convert function" << endl;
    }
    (*X).index[d1] = nnz; // use vals[index[i]:index[i+1]-1] to access i-th user
    R.clear_space();
	cout << "will return from convert " << (*X).nnz << endl;
    return X;
}

// implement a function to calculate lambda * matrix
// returns a matrix

mat_t copy_mat_t(mat_t& V, double lambda=1.0) {
	long d1 = static_cast<long>(V.size());
	int r = static_cast<int>(V[0].size());
	mat_t g(d1, vec_t(r));
	for (long i = 0; i < d1; ++i) {
		for (int j = 0; j < r; ++j) {
			g[i][j] = V[i][j] * lambda;
		}
	}
	return g;
}

// implement a function to calculate c * vector (c is some constant)
vec_t copy_vec_t(const vec_t& g, double c=1.0) {
	long n = static_cast<long>(g.size());
	vec_t res(g.size());
	for (long i = 0; i < n; ++i) {
		res[i] = g[i] * c;
	}
	return res;
}

// implement a function to update j-th row of matrix by adding some constant * i-th row of another matrix, both matrix of type mat_t
// we want g[j,:] += c * U[i,:]
void update_mat_add_vec(const vec_t& ui, double c, long j, mat_t& g) {
	long r = static_cast<long>(g[0].size());
	for (long k = 0; k < r; ++k) {
		g[j][k] += c * ui[k];
	}
	return;
}

// function to update subrange of a vec by adding another vector
void update_vec_subrange(const vec_t& ui, double c, vec_t& Ha, long Ha_start, long Ha_end) {
	long n = static_cast<long>(ui.size());
	assert(n == (Ha_end - Ha_start + 1));
	for (long i = 0; i < n; ++i) {
		Ha[Ha_start + i] += c * ui[i];
	}
	return;
}


// implement a function to compute c1 * vec1 + c2 * vec2
// trade-off: copying takes a little more time over reference
// BUT compiler will do optimization on copy elision though
// and function will be more general 
vec_t add_vec_vec(vec_t& g1, vec_t& g2, double c1=1.0, double c2=1.0) {
	assert(g1.size() == g2.size());
	vec_t res(g1.size());
	long n = static_cast<long>(g1.size());
	for (long i = 0; i < n; ++i) {
		res[i] = g1[i] * c1 + g2[i] * c2;
	}
	return res;
}


// implement a function to convert a matrix into a vector, namely vec() in julia
void vectorize_mat(const mat_t& g, vec_t& res) {
	// g is d2 by r
	long d2 = static_cast<long>(g.size());
	long r = static_cast<long>(g[0].size());
	res.resize(static_cast<unsigned int>(d2 * r));
	long res_size = static_cast<long>(res.size());
	assert(d2 * r == res_size);
	long cc = 0;
	for (long i = 0; i < d2; ++i) {
		for (long j = 0; j < r; ++j) {
			res[cc++] = g[i][j];
		}
	}
	return;
}


// implement dot product of vector and subrange of another vector
// requires vector and sub-vector to be of the same length
double vec_prod_array(const vec_t& ui, const vec_t& a, long a_start, long a_end) {
	double res = 0.0;
	long n = static_cast<long>(ui.size());
	assert(n == (a_end - a_start + 1));
	for (long i = 0; i < n; ++i) {
		res += ui[i] * a[a_start + i];
	}
	return res;
}

// implement matrix substract vector function, where vector is vectorized from same size matrix
// and requires vectorized is in the same order as vectorize_mat()
void mat_substract_vec(const vec_t& delta, double s, mat_t& V) {
	long d2 = static_cast<long>(V.size());
	long r = static_cast<long>(V[0].size());
	long delta_size = static_cast<long>(delta.size());
	assert(d2 * r == delta_size);
	long cc = 0;
	for (long i = 0; i < d2; ++i) {
		for (long j = 0; j < r; ++j) {
			V[i][j] -= s * delta[cc++];
		}
	}
	assert(cc == delta_size);
	return;
}







