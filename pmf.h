#ifndef _PMF_H_
#define _PMF_H_

#include "util.h"

enum {CCDR1, PCR, PCRPP};
enum {BOLDDRIVER, EXPDECAY};

class parameter {
	public:
		int solver_type;
		int k;
		int threads;
		int maxiter, maxinneriter;
		double lambda;
		double rho;
		double eps;						// for the fundec stop-cond in ccdr1
		double eta0, betaup, betadown;  // learning rate parameters used in DSGD
		int lrate_method, num_blocks; 
		int do_predict, verbose;
		int do_nmf;  // non-negative matrix factorization
		
		// liwei
		double stepsize;
		int ndcg_k;

		parameter() {
			// liwei
			double stepsize = 1.0;
			int ndcg_k = 10;

			solver_type = CCDR1;
			k = 10;
			rho = 1e-3;
			maxiter = 5;
			maxinneriter = 5;
			lambda = 0.1;
			threads = 4;
			eps = 1e-3;
			eta0 = 1e-3; // initial eta0
			betaup = 1.05;
			betadown = 0.5;
			num_blocks = 30;  // number of blocks used in dsgd
			lrate_method = BOLDDRIVER;
			do_predict = 0;
			verbose = 0;
			do_nmf = 0;
		}
};


extern "C" {
void ccdr1(smat_t &R, mat_t &W, mat_t &H, testset_t &T, parameter &param);
void pcr(smat_t &X, mat_t &U, mat_t &V, parameter &param);
}


#endif
