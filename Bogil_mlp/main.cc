/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "mlp.h"


using namespace std;

void print_usage(char *exec) {
    cout << "Usage: " << exec                               					<< endl
         << "       -config      <required: mlp config file>"    				<< endl
         << "       -test        <required: mlp test file>"      				<< endl
		 << "       -test_label  <required: mlp test label file>" 				<< endl
         << "       -train       <optional: mlp training file> and <integer value of starting point>"  				<< endl
		 << "       -train_label <optional: mlp training label file>" 			<< endl
         << "       -weight      <optional: mlp weight file to skip training>"	<< endl;
    exit(0);
}

int main(int argc, char **argv) {
    // Check # of input arguments.
    if(argc < 2) { print_usage(argv[0]); }

    // Parse input arguments.
    string config_file_name, weight_file_name;
    string test_file_name, train_file_name;
	string test_label_file_name, train_label_file_name;
	unsigned train_from;

    for(int i = 1; i < argc; i++) {
        if(!strcasecmp(argv[i], "-config")) {
            config_file_name = argv[++i];
        }
        else if(!strcasecmp(argv[i], "-test")) {
            test_file_name = argv[++i];
        }
		else if(!strcasecmp(argv[i], "-test_label")) {
			test_label_file_name = argv[++i];
		}
        else if(!strcasecmp(argv[i], "-train")) {
            train_file_name = argv[++i];
			train_from = atoi(argv[++i]);
        }
		else if(!strcasecmp(argv[i], "-train_label")) {
			train_label_file_name = argv[++i];
		}
        else if(!strcasecmp(argv[i], "-weight")) {
            weight_file_name = argv[++i];
		}
        else {
            cout << "Error: unknown option " << argv[i] << endl;
            exit(1);
        }
    }

    if(!config_file_name.size() || !test_file_name.size() || !test_label_file_name.size() ||
	   (!weight_file_name.size() && (!train_file_name.size() || !train_from  || !train_label_file_name.size()))) {
        print_usage(argv[0]);
    }

    mlp_t *mlp = new mlp_t(config_file_name, test_file_name, test_label_file_name,
                           train_file_name, train_from, train_label_file_name, weight_file_name);

    mlp->initialize();
	mlp->init_weights();
//	mlp->load_weights();
//	mlp->read_train_file();
//	mlp->read_train_label_file();
//	mlp->train(0.001, 0.9);
	mlp->read_test_file();
	mlp->read_test_label_file();
	mlp->test();
	mlp->clear();
    delete mlp;

    return 0;
}
