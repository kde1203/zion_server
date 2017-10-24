// Software: Testing Artificial Neural Network for MNIST database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2015 (c). All rights reserved.

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <pthread.h>

using namespace std;

// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "testing-report.dat";

// Number of testing samples
const int nTesting = 10000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 84; 
const int n3 = 10; // Ten classes: 0 - 9

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1];

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2];


int nCorrect = 0;
    

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	// Details
	cout << "*************************************************" << endl;
	cout << "*** Testing Neural Network for MNIST database ***" << endl;
	cout << "*************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "Testing image data: " << testing_image_fn << endl;
	cout << "Testing label data: " << testing_label_fn << endl;
	cout << "No. testing sample: " << nTesting << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 0; i < n1; i++) 
        w1[i] = new double [n2];
    
	// Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 0; i < n2; i++) 
        w2[i] = new double [n3];
}

// +----------------------------------------+
// | Load model of a trained Neural Network |
// +----------------------------------------+

void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);
	
	// Input layer - Hidden layer
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
			file >> w1[i][j];
		}
    }
	
	// Hidden layer - Output layer
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; j++) {
			file >> w2[i][j];
		}
    }
	
	file.close();
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

void perceptron(double *out1, double *out3) {

	double *in2, *out2, *in3;
	in2  = new double [n2];
    out2 = new double [n2];
    in3  = new double [n3];

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            in2[j] += out1[i] * w1[i][j];	
		}
	}

    for (int i = 0; i < n2; i++) 
		out2[i] = sigmoid(in2[i]);
	
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n3; j++) {
            in3[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 0; i < n3; i++) 
		out3[i] = sigmoid(in3[i]);
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input(char *s_image, double *out1) {
	// Reading image
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            if (s_image[height*j+i] == 0) 
				out1[i+j*height]=0;
			else out1[i+j*height]=1;
        }
	}
}

void *distribute(void *param) {

	int id = *(char *)param;
	int NUM_threads = *((char *)param+1);
	char *s_image =(char *)param+2;
	char *s_label;

	if(id!=NUM_threads-1)
		s_label = (char *)param+2+784*(10000/NUM_threads);
	else s_label = (char *)param+2+784*(10000/NUM_threads + 10000%NUM_threads);
	
	double *out1, *in2, *out2, *in3, *out3;

	int nThreadCorrect = 0; 

	out1 = new double [n1];
	out3 = new double [n3];
	double expected[n3]={0};

	if(id != NUM_threads-1){
		for(int sample = 0; sample < 10000/NUM_threads; sample++){

			input(s_image + 784*sample, out1);
			int label = s_label[sample];
			perceptron(out1, out3);

			int predict = 0;
	        for (int i = 1; i < n3; i++) {
				if (out3[i] > out3[predict]) 
					predict = i;
			}

			for(int i = 0; i < n3; i++)
				expected[i] = 0;
			expected[label] = 1;
			if (label == predict) 
				nThreadCorrect++;
		}
	}
	else if(id == NUM_threads-1){
		for(int sample = 0; sample < 10000/NUM_threads+10000%NUM_threads; sample++){

			input(s_image + 784*sample, out1);

			int label = s_label[sample];

			perceptron(out1, out3);

			int predict = 0;
	        for (int i = 1; i < n3; i++) {
				if (out3[i] > out3[predict]) 
					predict = i;
			}

			for(int i = 0; i < n3; i++)
				expected[i] = 0;
			expected[label] = 1;
			if (label == predict) 
				nThreadCorrect++;
		}
	}
	nCorrect += nThreadCorrect;

}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {

	if(argc!=2){
		printf("natural number is needed: ex) ./file_name ''integer''\n");
		return 0;
	}
	if(atoi(argv[1])<=0){
		printf("the number should be bigger than 0\n");
		return 0;
	}

	about();

    report.open(report_fn.c_str(), ios::out);
    image.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

		
	// Neural Network Initialization
    init_array(); // Memory allocation
    load_model(model_fn); // Load model (weight matrices) of a trained Neural Network

	int NUM_threads = atoi(argv[1]);

	char** thread_data; // 1 : thread amount, 2: thread number,  3: image size, 4: label size.
	thread_data = new char*[NUM_threads]; 
	for(int i=0; i<NUM_threads; i++){
		thread_data[i]=new char[787*(10000/NUM_threads)];	
	}	

	image.seekg(16, ios::beg);
	label.seekg(8, ios::beg);

	for(int i=0; i<NUM_threads; i++){
		thread_data[i][0]=i;
		thread_data[i][1]=NUM_threads;
		if(i!=NUM_threads-1){
			image.read(&thread_data[i][2], 784*(10000/NUM_threads));
			label.read(&thread_data[i][2+784*(10000/NUM_threads)], 10000/NUM_threads);
		}
		else {
			image.read(&thread_data[i][2], 784*(10000/NUM_threads+10000%NUM_threads));
			label.read(&thread_data[i][2+784*(10000/NUM_threads+10000%NUM_threads)], 10000/NUM_threads+10000%NUM_threads);
		}
	}
	pthread_t tid[NUM_threads];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	
	for(int i=0; i<NUM_threads; i++){
		pthread_create(&tid[thread_data[i][0]], &attr, distribute, (void*)&thread_data[i][0]);
	}
	for(int i=0; i<NUM_threads; i++){
		pthread_join(tid[thread_data[i][0]], NULL);
	}

	


	// Summary
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
    
    return 0;
}


