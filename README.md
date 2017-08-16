# Large-scale Collaborative Ranking in Near-Linear Time
## Announcement:
- The paper has been accepted for oral presentation (8.5% acceptance rate) by KDD’17, Halifax, Nova Scotia, Canada (http://www.kdd.org/kdd2017/accepted-papers).
- I will give an oral presentation about this work at Halifax sometime between August 13 - 17, 2017 (exact date and time to be announced later).
- You can cite the work as 

> Large-scale Collaborative Ranking in Near-Linear Time , Liwei Wu, Cho-Jui Hsieh, James Sharpnack. To appear in ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.

## Description

The code is modified from LIBPMF (https://www.cs.utexas.edu/~rofuyu/libpmf/).
We extend the codebase for LIBPMF to collaborative ranking. 

## Platform

Unix system with g++.

## Compilation

$ make

## Quick start

$ ./omp-pmf-train ml1m model
$ ./omp-pmf-predict ml1m/test.ratings model predicted_result 

## Data format

The input format is the same with LIBPMF. 
The input format of the training data is a directory containing a file called "meta", a file storing training ratings, and a file storing test ratings. 

    "meta" file contains three lines:
          1st: m n
          2nd: num_training_ratings training_file_name
          3rd: num_test_ratings test_file_name

See ml1m/ for a concrete example.

## Instructions on how to run the code:

Run each program without arguments to show the detailed usage: 

$ ./omp-pmf-train

	Usage: omp-pmf-train [options] data_dir [model_filename]
	options:
	    -s type : set type of solver (default 2)
		     1 -- PirmalCR
	         2 -- PrimalCR++
	    -k rank : set the rank (default 10)
		-n threads : set the number of threads (default 4)
		-l lambda : set the regularization parameter lambda (default 5000)
		-t max_iter: set the number of iterations (default 10)
		-p do_predict: compute training/testing error & NDCG at each iteration or not (default 1)

$ ./omp-pmf-predict

	Usage: omp-pmf-predict test_file model output_file

