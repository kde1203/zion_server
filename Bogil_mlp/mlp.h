/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <stdint.h>
#include <string>
#include <vector>

typedef uint8_t data_type_t;

// MLP layer types
enum MLP_LAYERS { NONE = 0, CONV, POOL, CLASS, NUM_LAYER_TYPES };

// MLP class
class mlp_t {
public:
	mlp_t(std::string m_config_file_name, std::string m_test_file_name,
		  std::string m_test_label_file_name, std::string m_train_file_name, unsigned train_from,
		  std::string m_train_label_file_name, std::string m_weight_file_name);		// MLP constructor
	virtual ~mlp_t();                           // MLP destructor

   	void initialize();                          // Initialize MLP parameters
	void clear();
	void init_weights();
	void load_weights();
	void save_weights();
    int  big_to_little_endian(int x);
	uint8_t bin_to_dec(uint8_t x);
	unsigned big_to_little_endian_unsigned(unsigned x);
	void read_test_file();
	void read_train_file();
	void read_test_label_file();
	void read_train_label_file();
	void calculate(float *output, float *input, unsigned num_neurons_output_layer, unsigned num_neurons_input_layer, unsigned count);
	void forward();
	unsigned get_output(float *arr);
	void train(float learningrate, float momentum);
	void test();
	float drelu(float num);

private:
    std::string config_file_name;               // Configuration file name
    std::string test_file_name;                 // Test file for inferencing
	std::string test_label_file_name;			// Test label file for inferencing check
    std::string train_file_name;             	// Training data set
	std::string train_label_file_name;			// Training data labels
    std::string weight_file_name;               // Pre-trained weight file name
	unsigned train_from;
  // Starting training point
    unsigned num_layers;                        // # of layers including I/O
    unsigned image_width, image_length;         // # of data points in the width, length

	unsigned train_set_size, test_set_size;
	unsigned *num_neurons_per_layer;

	uint8_t *test_label_set;
	uint8_t *train_label_set;
	float **neurons;
	float **deltavalues;
	float **theta;
    uint8_t **test_set;
	uint8_t **train_set;
	float **weights;
};

