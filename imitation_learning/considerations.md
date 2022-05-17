## Considerations when designing the neural network


Since the problems all have varying-length input, depending on the number of tanks, we consider the following options to
put it into a neural network:

- we pad the input to a fixed length of sufficient size
- we use RNN's to allow for varying input lenghts

We feel like RNN's will make the problem too complicated, and we want to find a relation between the entire input and output, so we think
a padding will be the most fitting here.


The question then becomes what padding to use, and to what size to pad. 

ideas are:
- pad the input with -1 until we reach a certain length for the varying-length input items (such as processing times and moving times between tanks)


What size to pad until?
If we look at the literature there are a number of benchmark problems out there, such as Black Oxide 1 and Zinc. 
The largest of these problems is Zinc, which has 16 tanks.
The paper of Neil has as largest multiplier 5, namely in their random instance generation. (see paper)
We therefore suggest padding until 5*16 = 80, as with this input size we can cover most of the known problems, and their corresponding benchmark problems
80 is maybe too large, too many input params?, try 60, as BO1 has 12 tanks, considering multiplier of 5

Problem: how to smartly reduce number of input parameters? (ask Neil)