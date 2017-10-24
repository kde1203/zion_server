/***************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul, South Korea
 *****************************************************/

#include <fstream>
#include <iostream>
#include <libconfig.h++>
#include <random>
#include "mlp.h"
#include <math.h>

#define EPOCH 1 

using namespace std;
using namespace libconfig;

mlp_t::mlp_t(string m_config_file_name,
             string m_test_file_name,
			 string m_test_label_file_name,
             string m_train_file_name,
			 unsigned m_train_from,
			 string m_train_label_file_name,
             string m_weight_file_name) :
    config_file_name(m_config_file_name),
    test_file_name(m_test_file_name),
	test_label_file_name(m_test_label_file_name),
    train_file_name(m_train_file_name),
	train_from(m_train_from),
	train_label_file_name(m_train_label_file_name),
    weight_file_name(m_weight_file_name),
    image_width(0), image_length(0) {
}


mlp_t::~mlp_t() {
}

void mlp_t::clear(){
	delete [] num_neurons_per_layer;
	if(test_set){
		for(unsigned i = 0; i < test_set_size; i++){
			delete [] test_set[i];
		}
	}
	delete [] test_set;
	delete [] test_label_set;
	if(train_set){
		for(unsigned i = 0; i <train_set_size; i++){
			delete [] train_set[i];
		}
	}
	delete [] train_set;
	delete [] train_label_set;
	if(neurons){
		for(unsigned i = 0; i < num_layers; i++){
			delete [] neurons[i];
		}
	}
	delete [] neurons;
	if(weights){
		for(unsigned i = 0; i<num_layers-1; i++){
			delete [] weights[i];
		}
	}
	delete [] weights;
	if(deltavalues){
		for(unsigned i = 0; i<num_layers-1; i++){
			delete [] deltavalues[i];
		}
	}
	delete [] deltavalues;
	if(theta){
		for(unsigned i = 0; i<num_layers-1; i++){
			delete [] theta[i];
		}
	}
	delete [] theta;
}

void mlp_t::initialize() {
    // User libconfig to parse a configuration file.
    Config mlp_config;
    mlp_config.readFile(config_file_name.c_str());

    try {
        // Load image size and hidden layer array settings.
        Setting &s_image_size = mlp_config.lookup("image_size");
        Setting &s_num_neurons_hidden_layer = mlp_config.lookup("num_neurons_hidden_layer");

        // +2 means 1 for input and another 1 for output layer
		// num_layers = s_num_neurons_hidden_layer.getLength()+2;
/********************/		
		num_layers=3;
/*****************/
        // Load the # of neurons in the input layer.
        if(s_image_size.getLength() != 2) {
            cerr << "image_size must be defined [width, length]" << endl;
            exit(1);
        }
		num_neurons_per_layer = new unsigned[num_layers];
        // +1 is for a label.
        image_width = unsigned(s_image_size[0]);
        image_length = unsigned(s_image_size[1]);
		num_neurons_per_layer[0] = image_width*image_length;
        // Load the number of neurons in each hidden layer.
//		for(int i = 0; i < s_num_neurons_hidden_layer.getLength(); i++) {
//			num_neurons_per_layer[i+1] = unsigned(s_num_neurons_hidden_layer[i]);
//		}
/***********/
		num_neurons_per_layer[1] = unsigned(mlp_config.lookup("num_neurons_hidden_layer"));
/**********/
        // Load the number of neurons in the output layer.
		num_neurons_per_layer[num_layers-1] = unsigned(mlp_config.lookup("num_neurons_output_layer"));
        test_set_size = unsigned(mlp_config.lookup("test_set_size"));
		train_set_size = unsigned(mlp_config.lookup("train_set_size"));
        // Vector of weight matrix. "-1" means there are
        // no more weights from the output layer.
		test_label_set = new uint8_t[test_set_size];
        test_set = new uint8_t*[test_set_size];
        for(size_t i = 0; i < test_set_size; i++) {
            test_set[i] = new uint8_t[image_width*image_length];
        }
		train_label_set= new uint8_t[train_set_size];
		train_set = new uint8_t*[train_set_size];
		for(size_t i = 0; i < train_set_size; i++) {
			train_set[i] = new uint8_t[image_width*image_length];
		}
		neurons = new float*[num_layers];
		for(unsigned i = 0; i <num_layers; i++){
			neurons[i] = new float[num_neurons_per_layer[i]];
		}
		deltavalues = new float*[num_layers-1];
		for(unsigned i = 0; i < num_layers-1; i++){
			deltavalues[i] = new float[num_neurons_per_layer[i+1]*num_neurons_per_layer[i]];
		}
		weights = new float*[num_layers-1];
        for(unsigned i = 0; i < num_layers-1; i++) {
			weights[i] = new float[num_neurons_per_layer[i+1]*num_neurons_per_layer[i]];
        }
		theta = new float*[num_layers-1];
		for(unsigned i = 0; i < num_layers-1; i++){
			theta[i] = new float[num_neurons_per_layer[i+1]];
		}
        if(!weight_file_name.size()) { init_weights(); }
        else { load_weights(); }

    }
    catch(SettingNotFoundException e) {
        cout << "Error: " << e.getPath() << " is not defined in "
             << config_file_name << endl;
    }
    catch(SettingTypeException e) {
        cout << "Error: " << e.getPath() << " has incorrect type in "
             << config_file_name << endl;
    }
    catch(FileIOException e) {
        cout << "Error: " << config_file_name << " does not exist" << endl;
    }
    catch(ParseException e) {
        cout << "Error: Failed to parse line # " << e.getLine()
             << " in " << config_file_name << endl;
    }
}

void mlp_t::init_weights(){
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 0.01);
	    for(unsigned i = 0; i < num_layers-1; i++) {
        for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++) {
            for(unsigned k = 0; k < num_neurons_per_layer[i]; k++) {
                weights[i][j*num_neurons_per_layer[i]+k] = distribution(generator);
				deltavalues[i][j * num_neurons_per_layer[i] +k] = 0.0;
			}
		}
	}
}

void mlp_t::load_weights() {
    ifstream file_stream;
    file_stream.open(weight_file_name.c_str(), ios::in);

    if(!file_stream.is_open()) {
        cerr << "Error: failed to open " << weight_file_name << endl;
        exit(1);
    }
	for(unsigned i = 0; i < num_layers-1; i++){
		for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++){
			for(unsigned k =0; k < num_neurons_per_layer[i]; k++){
				file_stream >> weights[i][j*num_neurons_per_layer[i]+k];
				deltavalues[i][j * num_neurons_per_layer[i] +k] = 0.0;
			}
		}
	}
}

void mlp_t::save_weights(){
	ofstream file("weights.txt", ios::out);
	for(unsigned i = 0; i < num_layers-1; i++){
		for(unsigned j = 0; j < num_neurons_per_layer[i+1]; j++){
			for(unsigned k =0; k < num_neurons_per_layer[i]; k++){
				file << weights[i][j*num_neurons_per_layer[i] + k] << " ";
			}
		}
		cout << endl;
	}
}
	
int mlp_t::big_to_little_endian(int x) {
    int tmp = ((x << 8) & 0xFF00FF00) | ((x >> 8) & 0x00FF00FF);
    return ((tmp << 16) | (tmp >> 16));
}

void mlp_t::read_test_file(){
    int dummy;
	uint8_t *num;
	num = new uint8_t[test_set_size * image_length * image_width];
	ifstream file_stream;
    file_stream.open(test_file_name.c_str(), fstream::in|fstream::binary);
    if(!file_stream.is_open()) {
        cerr << "Error: failed to open " << test_file_name << endl;
        exit(1);
    }
	for(unsigned i = 0; i <test_set_size*image_width*image_length+4; i++){
		if(i <4){
			file_stream.read((char*)&dummy, sizeof(int));
		}
		else{
			file_stream.read((char*)&num[i-4],sizeof(uint8_t));
		}
	}
	for(unsigned i = 0; i < test_set_size; i++){
		for(unsigned j=0; j < image_length; j++){
			for(unsigned k =0; k <image_width; k++){
				test_set[i][j*image_width +k] = num[i*image_width*image_length + j*image_length + k];
			}		
		}
	}	
	delete [] num;
}

void mlp_t::read_train_file() {
	int dummy;
	uint8_t *num;
	num = new uint8_t[train_set_size * image_length * image_width];
	ifstream file_stream;
    file_stream.open(train_file_name.c_str(), fstream::in|fstream::binary);
    if(!file_stream.is_open()) {
        cerr << "Error: failed to open " << train_file_name << endl;
        exit(1);
    }
	for(unsigned i = 0; i <train_set_size*image_width*image_length+4; i++){
		if(i <4){
			file_stream.read((char*)&dummy, sizeof(int));
		}
		else{
			
			file_stream.read((char*)&num[i-4],sizeof(uint8_t));
		}
	}
	for(unsigned i = 0; i < train_set_size; i++){
		for(unsigned j=0; j < image_length; j++){
			for(unsigned k =0; k <image_width; k++){
				train_set[i][j*image_width +k] = num[i*image_width*image_length + j*image_length + k];
			}		
		}
	}
	delete [] num;
}

void mlp_t::read_test_label_file(){
	int dummy;
	ifstream file_stream;
	file_stream.open(test_label_file_name.c_str(), fstream::in|fstream::binary);
	if(!file_stream.is_open()){
		cerr << "Error: failed to open " << test_label_file_name << endl;
		exit(1);
	}
	for(unsigned i = 0; i < test_set_size+2; i++){
		if(i < 2){
			file_stream.read((char*)&dummy,sizeof(int));
		}
		else{
			file_stream.read((char*)&test_label_set[i-2],sizeof(uint8_t));
		}
	}
}	

void mlp_t::read_train_label_file(){
	int dummy;
	ifstream file_stream;
	file_stream.open(train_label_file_name.c_str(), fstream::in|fstream::binary);
	if(!file_stream.is_open()){
		cerr << "Error: failed to open " << train_label_file_name << endl;
		exit(1);
	}
	for(unsigned i = 0; i < train_set_size+2; i++){
		if(i < 2){
			file_stream.read((char*)&dummy,sizeof(int));
		}
		else{
			file_stream.read((char*)&train_label_set[i-2],sizeof(uint8_t));
		}
	}
}	

//calcOneLayer, getInnerProduct, sigmoid..
void mlp_t::calculate(float *output, float* input, unsigned num_neurons_output_layer, unsigned num_neurons_input_layer, unsigned count){
	float sum;
	for(unsigned i = 0; i < num_neurons_output_layer; i++){
		sum = 0.0;
		for(unsigned j = 0; j < num_neurons_input_layer; j++){
		sum += weights[count][i * num_neurons_input_layer + j] * input[j];
		}
		output[i] = 1.0/(1.0 + exp(-sum));
	}
}

void mlp_t::forward(){
	for(unsigned i = 0; i < num_layers-1; i++){
		calculate(neurons[i+1],neurons[i],num_neurons_per_layer[i+1],num_neurons_per_layer[i],i);		
	}	
}
/*************/
unsigned mlp_t::get_output(float *arr){
/*
	float max_val=-1e9;
	unsigned index;
	for(unsigned i = 0; i<10; i++){
		if(max_val < arr[i]){
			max_val=arr[i];
			index=i;
		}
	}
	return index;
*/
	return 0;
}
/***********/

void mlp_t::train(float learningrate, float momentum){
	unsigned *desired;
	float sum;
	desired = new unsigned[num_neurons_per_layer[num_layers-1]];
	for(unsigned i = train_from-1; i < train_set_size; i++){
		for(unsigned j = 0; j<num_neurons_per_layer[num_layers-1]; j++){
			if(j == train_label_set[i]){
				desired[j]=1.0;
			}
			else{
				desired[j]=0.0;
			}
		}
		for(unsigned j = 0; j < num_neurons_per_layer[0]; j++){
			neurons[0][j] = train_set[i][j];
		}
		for(unsigned m =0; m < EPOCH; m++){
			forward();
//backpropagation
			for(unsigned j = 0; j < num_neurons_per_layer[num_layers-1]; j++){
				theta[num_layers-2][j]=  neurons[num_layers-1][j] * (1-neurons[num_layers-1][j]) * ( desired[j]-neurons[num_layers-1][j]);
			}
			for(unsigned j = num_layers-2; j >0; j--){
				for(unsigned k=0; k< num_neurons_per_layer[j]; k++){
					sum = 0.0;
					for(unsigned l =0; l <num_neurons_per_layer[j+1]; l++){
						sum += theta[j][l]*weights[j][l*num_neurons_per_layer[j]+k];
					}
					theta[j-1][k] = neurons[j][k]*(1-neurons[j][k])*sum;
				}
			}
			for(unsigned j = 0; j<num_layers-1; j++){
				for(unsigned k =0; k < num_neurons_per_layer[j+1]; k++){
					for(unsigned l = 0; l < num_neurons_per_layer[j]; l++){
						weights[j][k*num_neurons_per_layer[j]+l] += momentum*deltavalues[j][k*num_neurons_per_layer[j]+l];
					}
				}
			}
			for(unsigned j = 0; j <num_layers-1; j++){
				for(unsigned k =0; k<num_neurons_per_layer[j+1]; k++){
					for(unsigned l = 0; l<num_neurons_per_layer[j]; l++){
						deltavalues[j][k*num_neurons_per_layer[j]+l] = learningrate*theta[j][k]*neurons[j][l];
						weights[j][k*num_neurons_per_layer[j]+l] +=deltavalues[j][k*num_neurons_per_layer[j]+l];
					}
				}
			}
		
		}
		cout << "train set #" << i+1 << endl;
		cout << "label is : " << (int) train_label_set[i] << endl;
		for (unsigned j=0; j < image_width; j++){
			for(unsigned k=0; k < image_length; k++){
				if(train_set[i][k+j*image_length]==0) cout << 0;
				else cout << 1;
			} cout << endl;
		} cout << endl;
		if(i%1000==0) save_weights();
	}
	delete [] desired;
}

void mlp_t::test(){
	unsigned correct = 0;
	unsigned output;
	float max_val;
	for(unsigned i = 0; i < test_set_size; i++){
		for(unsigned j = 0; j < num_neurons_per_layer[0]; j++){
			neurons[0][j]=test_set[i][j];
		}
		forward();
		max_val=0.0;
		output = 0;
		for(unsigned j =0; j < num_neurons_per_layer[num_layers-1]; j++){
			if(max_val < neurons[num_layers-1][j]){
				max_val = neurons[num_layers-1][j];
				output = j;
			}
		}
		if(test_label_set[i] == output){correct += 1;}
		else {
			cout << (unsigned)test_label_set[i] << " vs " << output << endl;
		}
	}
	cout << "correct ratio is : " << correct << endl;	
}
