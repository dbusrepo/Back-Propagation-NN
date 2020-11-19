#pragma once

#include "common.h"



struct BackPropParameters {
	size_t maxGenerations;
	size_t numHiddenNodes;
	RealType eta;    // learning rate
	RealType alpha;  // momentum
	RealType lambda; // weight decay
	RealType errToll;
	void updateLearningRate();
	void updateMomentumRate();
};

// Backpropagation for a 3 layers neural network (input-hidden-output
class BackProp {
public:
	BackProp();

	void runBatch();

private:

	void allocateWeightBiasMem();
	void setRandomWeightBias();
	void computeNetInputs(RealType *netInputs, RealType *weights, int nNodes, RealType *inputs, int nInputs, RealType *bias);

	// algorithm parameters
	BackPropParameters *parameters;

	// #training patterns
	size_t numTrPatterns;

	// #neurons for each layer
	size_t nInputNodes, nHiddenNodes, nOutputNodes;

	// outputs values for each pattern, for each layer
	RealType *outInput, *outHidden, *outOutput;

	// pattern target values
	RealType *target;


	// weights matrices. number of rows: #neurons of cur layer, number of columns: #neurons previous layer
	RealType *weightsHidden, *weightsOutput;
	// weight changes
	RealType *dWeightsHidden, *dWeightsOutput;

	// bias vectors
	RealType *biasHidden, *biasOutput;

	// bias changes
	RealType *dBiasHidden, *dBiasOutput;


};