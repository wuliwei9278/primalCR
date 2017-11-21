#include "util.h"
#include "pmf.h"
#include <fstream>
#include <cstring>

bool with_weights;

void exit_with_help()
{
	printf(
	"Usage: omp-pmf-train [options] data_dir [model_filename]\n"
	"options:\n"
	"    -s type : set type of solver (default 2)\n"    
//	"    	 0 -- CCDR1 with fundec stopping condition\n"    
	"    	 1 -- PirmalCR\n"    
	"    	 2 -- PrimalCR++\n"    
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 5000)\n"    
	"    -t max_iter: set the number of iterations (default 10)\n"    
//	"    -T max_iter: set the number of inner iterations used in CCDR1 (default 5)\n"    
//	"    -e epsilon : set inner termination criterion epsilon of CCDR1 (default 1e-3)\n"     
	"    -p do_predict: compute training/testing error & NDCG at each iteration or not (default 1)\n"    
//	"    -q verbose: show information or not (default 0)\n"
	);
	exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	parameter param;   // default values have been set by the constructor 
	with_weights = false;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.solver_type = atoi(argv[i]);
				break;

			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'n':
				param.threads = atoi(argv[i]);
				break;

			case 'l':
				param.lambda = atof(argv[i]);
				break;

			case 'r':
				param.rho = atof(argv[i]);
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				param.maxinneriter = atoi(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				param.eta0 = atof(argv[i]);
				break;

			case 'B':
				param.num_blocks = atoi(argv[i]);
				break;

			case 'm':
				param.lrate_method = atoi(argv[i]);
				break;

			case 'u':
				param.betaup = atof(argv[i]);
				break;

			case 'd':
				param.betadown = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'N':
				param.do_nmf = atoi(argv[i]) == 1? true : false;
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict!=0) 
		param.verbose = 1;

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') 
			*p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	return param;
}


void run_ccdr1(parameter &param, const char* input_file_name, const char* model_file_name=NULL){
	smat_t R;
	mat_t W,H;
	testset_t T;

	FILE *model_fp = NULL;

	if(model_file_name) {
		model_fp = fopen(model_file_name, "wb");
		if(model_fp == NULL)
		{
			fprintf(stderr,"can't open output file %s\n",model_file_name);
			exit(1);
		}
	}

	load(input_file_name,R,T, with_weights);
	// W, H  here are k*m, k*n
	initial_col(W, param.k, R.rows);
	initial_col(H, param.k, R.cols);

	//printf("global mean %g\n", R.get_global_mean());
	//printf("global mean %g W_0 %g\n", R.get_global_mean(), norm(W[0]));
	puts("starts!");
	double time = omp_get_wtime();
	ccdr1(R, W, H, T, param);
	printf("Wall-time: %lg secs\n", omp_get_wtime() - time);

	if(model_fp) {
		save_mat_t(W,model_fp,false);
		save_mat_t(H,model_fp,false);
		fclose(model_fp);
	}
	return ;
}

void run_pcr(parameter &param, const char* input_file_name, const char* model_file_name=NULL){
	smat_t X;
	testset_t T;  // declared to pass into load, which has to take T, not used otherwise
	mat_t U, V;
	FILE *model_fp = NULL;
	if(model_file_name) {
        model_fp = fopen(model_file_name, "wb");
        if(model_fp == NULL)
        {
            fprintf(stderr,"can't open output file %s\n",model_file_name);
            exit(1);
        }
    }

	load(input_file_name, X, T, false);
	// param.k for rank (r in julia)
	// param.lambda (lambda in julia)
	// param.stepsize, param.ndcg_k, ...
	
	//U = read_initial("initial_U");
	//V = read_initial("initial_V");
	// k by n
	initial(U, X.rows, param.k);
	//cout << U.size() << " and " << U[0].size() << endl;
	initial(V, X.cols, param.k);
	cout << "the rank is " << param.k << endl;
	cout << "the number of rows is " << X.rows << " and the number of cols is " << X.cols << endl;
	cout << "nnz: " << X.nnz << endl;	
	
    double time = omp_get_wtime();
    pcr(X, U, V, T, param);
    printf("Wall-time: %lg secs\n", omp_get_wtime() - time);
	
	cout << "U matrix of size " << U.size() << ", " << param.k << endl;
    ofstream myfile;
    myfile.open("U" + to_string(static_cast<int>(param.lambda)) + ".txt");
    for (long i = 0; i < U.size(); ++i) {
        for (long j = 0; j < param.k; ++j) {
            myfile << U[i][j];
            if (j < param.k - 1) myfile << " ";
            else myfile << "\n";
        }
    }
    myfile.close();
    myfile.open("V" + to_string(static_cast<int>(param.lambda)) + ".txt");
    cout << "V matrix of size " << V.size() << ", " << param.k << endl;
    for (long i = 0; i < V.size(); ++i) {
        for (long j = 0; j < param.k; ++j) {
            myfile << V[i][j];
            if (j < param.k - 1) myfile << " ";
            else myfile << "\n";
        }
    }
    myfile.close();

    if(model_fp) {
		mat_t UT(param.k, vec_t(U.size()));
		for ( int ii=0 ; ii<U.size() ; ii++ )
			for ( int jj=0 ; jj<param.k ; jj++ )
				UT[jj][ii] = U[ii][jj];
		mat_t VT(param.k, vec_t(V.size()));
		for ( int ii=0 ; ii<V.size() ; ii++ )
			for ( int jj=0 ; jj<param.k ; jj++ )
				VT[jj][ii] = V[ii][jj];
        save_mat_t(UT,model_fp,false);
        save_mat_t(VT,model_fp,false);
        fclose(model_fp);
    }
    
	return;
}


void run_pcrpp(parameter &param, const char* input_file_name, const char* model_file_name=NULL){
	smat_t X;
    testset_t T;  // declared to pass into load, which has to take T, not used otherwise
    mat_t U, V;
    FILE *model_fp = NULL;
    if(model_file_name) {
        model_fp = fopen(model_file_name, "wb");
        if(model_fp == NULL)
        {
            fprintf(stderr,"can't open output file %s\n",model_file_name);
            exit(1);
        }
    }

    load(input_file_name, X, T, false);
//	U = read_initial("initial_U");
//	V = read_initial("initial_V");
	initial(U, X.rows, param.k);
	//cout << U.size() << " and " << U[0].size() << endl;
	initial(V, X.cols, param.k);

	cout << "the rank is " << param.k << endl;
    cout << "the number of rows is " << X.rows << " and the number of cols is " << X.cols << endl;
    cout << X.nnz << endl;
    cout << "starts!" << endl;
	double time = omp_get_wtime();
    pcrpp(X, U, V, T, param);
    printf("Wall-time: %lg secs\n", omp_get_wtime() - time);
    cout << "U matrix of size " << U.size() << ", " << param.k << endl;
    ofstream myfile;
    myfile.open("U.txt");
    for (long i = 0; i < U.size(); ++i) {
        for (long j = 0; j < param.k; ++j) {
            myfile << U[i][j];
            if (j < param.k - 1) myfile << " ";
            else myfile << "\n";
        }
    }
    myfile.close();
    myfile.open("V.txt");
    cout << "V matrix of size " << V.size() << ", " << param.k << endl;
    for (long i = 0; i < V.size(); ++i) {
        for (long j = 0; j < param.k; ++j) {
            myfile << V[i][j];
            if (j < param.k - 1) myfile << " ";
            else myfile << "\n";
        }
    }
    myfile.close();

    if(model_fp) {
		mat_t UT(param.k, vec_t(U.size()));
		for ( int ii=0 ; ii<U.size() ; ii++ )
			for ( int jj=0 ; jj<param.k ; jj++ )
				UT[jj][ii] = U[ii][jj];
		mat_t VT(param.k, vec_t(V.size()));
		for ( int ii=0 ; ii<V.size() ; ii++ )
			for ( int jj=0 ; jj<param.k ; jj++ )
				VT[jj][ii] = V[ii][jj];

        save_mat_t(UT,model_fp,false);
        save_mat_t(VT,model_fp,false);
        fclose(model_fp);
    }

	
	return;
}

int main(int argc, char* argv[]){
	char input_file_name[1024];
	char model_file_name[1024];
	parameter param = parse_command_line(argc, argv, input_file_name, model_file_name); 

	switch (param.solver_type){
		case CCDR1:
			run_ccdr1(param, input_file_name, model_file_name);
			break;
		case PCR:
			run_pcr(param, input_file_name, model_file_name);
			break;
		case PCRPP:
			run_pcrpp(param, input_file_name, model_file_name);
			break;
		default:
			fprintf(stderr, "Error: wrong solver type (%d)!\n", param.solver_type);
			break;
	}
	return 0;
}

