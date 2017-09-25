#include "BackProp.h"
#include "tanh.hpp"

BackProp::BackProp()
{

}

void BackProp::allocateWeightBiasMem() {
	weightsHidden = new RealType[nHiddenNodes * nInputNodes];
	weightsOutput = new RealType[nOutputNodes * nHiddenNodes];
	dWeightsHidden = new RealType[nHiddenNodes * nInputNodes];
	dWeightsOutput = new RealType[nOutputNodes * nHiddenNodes];

	biasHidden = new RealType[nHiddenNodes];
	biasOutput = new RealType[nOutputNodes];
	dBiasHidden = new RealType[nHiddenNodes];
	dBiasOutput = new RealType[nOutputNodes];
}

void BackProp::runBatch()
{
	allocateWeightBiasMem();

	// init weights and biases, also weights changes (for momentum term)

	outInput = new RealType[numTrPatterns * nInputNodes];
	outHidden = new RealType[numTrPatterns * nHiddenNodes];
	outOutput = new RealType[numTrPatterns * nOutputNodes];
	target = new RealType[numTrPatterns * nOutputNodes];

	// load input values for each pattern in outInput

	// delta values for each pattern, for hidden/output layer
	RealType *deltaHidden = new RealType[numTrPatterns * nHiddenNodes];
	RealType *deltaOutput = new RealType[numTrPatterns * nOutputNodes];

	// pattern errors (t-a)
	RealType *patternError = new RealType[numTrPatterns];

	size_t maxGenerations = parameters->maxGenerations;

	for (size_t nIter = 0; nIter != maxGenerations; ++nIter) {

		// introduce patternArray, and use outInput as a view on a batch on it
		// batchSize, numBatches = numTrPatterns / batchSize
		// for (int batch = 0;. ..) { for (int patt = 0; patt < pattPerBatch) {}}

		for (size_t p = 0; p != numTrPatterns; ++p) {

		  // forward pass

			// hidden layer
			RealType *input = outInput + p * nInputNodes;
			RealType *outHid = outHidden + p * nHiddenNodes;
			RealType *deltaHid = deltaHidden + p * nHiddenNodes;
			
			computeNetInputs(outHid, weightsHidden, nHiddenNodes, outInput, nInputNodes, biasHidden);

			for (size_t h = 0; h != nHiddenNodes; ++h) {
				outHid[h] = tanh_libc(outHid[h]);
				deltaHid[h] = 0;
			}

			// output layer
			RealType *output = outOutput + p * nOutputNodes;

			computeNetInputs(output, weightsOutput, nOutputNodes, outHid, nHiddenNodes, biasOutput);

			for (size_t j = 0; j != nOutputNodes; ++j) {
				output[j] = tanh_libc(output[j]);
			}

			// backward pass: compute deltas for current pattern

			// output layer
			RealType *deltaOut = deltaOutput + p * nOutputNodes;
			RealType *targetValues = target + p * nOutputNodes;
			RealType *nodeWeights = weightsOutput;

			RealType error = 0;

			for (size_t j = 0; j != nOutputNodes; ++j, nodeWeights += nHiddenNodes) {
				RealType dvalue = targetValues[j] - output[j];
				error += dvalue  * dvalue;
				RealType temp = deltaOut[j] = dvalue * (1 - output[j] * output[j]);
				for (size_t h = 0; h != nHiddenNodes; ++h) {
					deltaHid[h] += temp * nodeWeights[h];
				}
			}

			patternError[p] += error / nOutputNodes;

			// hidden layer
			for (size_t h = 0; h != nHiddenNodes; ++h) {
				deltaHid[h] *= (1 - outHid[h] * outHid[h]); // fprime(outHid[h])
			}

		} // end cycle on patterns

		// weights update

		RealType eta = parameters->eta;
		RealType alpha = parameters->alpha;

		// adapt weights hidden->output
		RealType *weightsToJ = weightsOutput;
		RealType *dWeightsToJ = dWeightsOutput;
		for (size_t j = 0; j != nOutputNodes; ++j, weightsToJ += 1 + nHiddenNodes, dWeightsToJ += 1 + nHiddenNodes) {
			RealType sum = 0;
			
			// update bias to j
			for (size_t p = 0; p != numTrPatterns; ++p) {
				sum += deltaOutput[p * nOutputNodes + j];
			}

			RealType dw = eta * sum + alpha * dWeightsToJ[nHiddenNodes];
			weightsToJ[nHiddenNodes] += dw; // update bias of j
			dWeightsToJ[nHiddenNodes] = dw;

			// update weights to j
			for (size_t h = 0; h != nHiddenNodes; ++h) {
				RealType sum = 0;

				for (size_t p = 0; p != numTrPatterns; ++p) {
					sum += deltaOutput[p * nOutputNodes + j] * outHidden[p * nHiddenNodes + h];
				}

				dw = eta * sum + alpha * dWeightsToJ[h];
				weightsToJ[h] += dw; // update weight h-j
				dWeightsToJ[h] = dw;
			}
			
		}

		// adapt weights input->hidden
		RealType *weightsToH = weightsHidden;
		RealType *dWeightsToH = dWeightsHidden;
		for (size_t h = 0; h != nHiddenNodes; ++h, weightsToH += 1 + nInputNodes, dWeightsToH += 1 + nInputNodes) {

			RealType sum = 0;

			// update bias to h
			for (size_t p = 0; p != numTrPatterns; ++p) {
				sum += deltaHidden[p * nHiddenNodes + h];
			}

			RealType dw = eta * sum + alpha * dWeightsToH[nInputNodes];
			weightsToH[nInputNodes] += dw; // update bias of h
			dWeightsToH[nInputNodes] = dw;

			// update weights to h
			for (size_t i = 0; i != nInputNodes; ++i) {
				RealType sum = 0;

				for (size_t p = 0; p != numTrPatterns; ++p) {
					sum += deltaHidden[p * nHiddenNodes + h] * outInput[p * nInputNodes + i];
				}

				RealType dw = eta * sum + alpha * dWeightsToH[i];
				weightsToH[i] += dw; // update weight i-h
				dWeightsToH[i] = dw;
			}
		}

	}
}


void BackProp::setRandomWeightBias()
{

}

void BackProp::computeNetInputs(RealType *netInputs, RealType *weights, int nNodes, RealType *inputs, int nInputs, RealType *bias)
{
	// weights: nNodes x nInputs
	// bias: nInputs

	RealType *ptrWeights = weights;
	for (size_t i = 0; i != nNodes; ++i) {
		RealType sum = 0;
		for (size_t j = 0; j != nInputs; ++j) {
			sum += ptrWeights[j] * inputs[j];
		}
		netInputs[i] = sum;
		ptrWeights += nInputs;
	}

	for (size_t i = 0; i != nNodes; ++i) {
		netInputs[i] += bias[i];
	}
}

void BackPropParameters::updateLearningRate()
{
}

void BackPropParameters::updateMomentumRate()
{
}

/*

// forward pass

// hidden layer
RealType *input = outInput + p * nInputNodes;
RealType *outHid = outHidden + p * nHiddenNodes;
RealType *deltaHid = deltaHidden + p * nHiddenNodes;
RealType *nodeWeights = weightsHidden;
for (size_t h = 0; h != nHiddenNodes; ++h, nodeWeights += nInputNodes + 1) {
RealType sum = nodeWeights[nInputNodes];
for (size_t i = 0; i != nInputNodes; ++i) {
sum += nodeWeights[i] * input[i];
}
outHid[h] = sum; //fact(sum)
deltaHid[h] = 0;
}

// output layer
RealType *output = outOutput + p * nOutputNodes;
nodeWeights = weightsOutput;
for (size_t j = 0; j != nOutputNodes; ++j, nodeWeights += nHiddenNodes + 1) {
RealType sum = nodeWeights[nHiddenNodes];
for (size_t i = 0; i != nHiddenNodes; ++i) {
sum += nodeWeights[i] * outHid[i];
}
output[j] = sum; //fact(sum)
}


*/

/*

// adapt weights hidden->output
RealType *weightsToJ = weightsOutput;
RealType *dWeightsToJ = dWeightsOutput;
for (size_t j = 0; j != nOutputNodes; ++j, weightsToJ += 1 + nHiddenNodes, dWeightsToJ += 1 + nHiddenNodes) {
RealType sum = 0;

// update bias to j
for (size_t p = 0; p != numTrPatterns; ++p) {
sum += deltaOutput[p * nOutputNodes + j];
}

RealType dw = eta * sum + alpha * dWeightsToJ[nHiddenNodes];
weightsToJ[nHiddenNodes] += dw; // update bias of j
dWeightsToJ[nHiddenNodes] = dw;

// update weights to j
for (size_t h = 0; h != nHiddenNodes; ++h) {
RealType sum = 0;

for (size_t p = 0; p != numTrPatterns; ++p) {
sum += deltaOutput[p * nOutputNodes + j] * outHidden[p * nHiddenNodes + h];
}

dw = eta * sum + alpha * dWeightsToJ[h];
weightsToJ[h] += dw; // update weight h-j
dWeightsToJ[h] = dw;
}

}

// adapt weights input->hidden
RealType *weightsToH = weightsHidden;
RealType *dWeightsToH = dWeightsHidden;
for (size_t h = 0; h != nHiddenNodes; ++h, weightsToH += 1 + nInputNodes, dWeightsToH += 1 + nInputNodes) {

RealType sum = 0;

// update bias to h
for (size_t p = 0; p != numTrPatterns; ++p) {
sum += deltaHidden[p * nHiddenNodes + h];
}

RealType dw = eta * sum + alpha * dWeightsToH[nInputNodes];
weightsToH[nInputNodes] += dw; // update bias of h
dWeightsToH[nInputNodes] = dw;

// update weights to h
for (size_t i = 0; i != nInputNodes; ++i) {
RealType sum = 0;

for (size_t p = 0; p != numTrPatterns; ++p) {
sum += deltaHidden[p * nHiddenNodes + h] * outInput[p * nInputNodes + i];
}

RealType dw = eta * sum + alpha * dWeightsToH[i];
weightsToH[i] += dw; // update weight i-h
dWeightsToH[i] = dw;
}
}

*/