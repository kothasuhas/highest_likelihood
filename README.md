# Highest Likelihood
Greedy decoding is not greedy enough - find the highest likelihood sequence of a sepcified length

### Set up 
`pip install torch transformers`
`python3 highest_likelihood.py --model_name {model} --num_tokens 188 --desired_length {length} --greedy ;`

The default model is pythia-70m
The length argument specifies how long the length should be.
`--greedy` turns on regular greedy decoding. Removing this flag is running highest likelihood instead.

Example logs can be found in the `logs` folder.