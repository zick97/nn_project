# Neural Network Project
## Learning to play Connect4 using a Recurrent Neural Network
This project focuses on the idea that the game of *Connect4* - as well as many other table or puzzle games - consists in just a sequence of moves. 
To be exact, each player realizes his/her own idea of this sequence, deciding which move to make in order to win the game.\
Thanks to this simple and straight-forward structure, I decided to evaluate the performance of various Machine Learning models at this poppular game, 
seeing them just as normal players. The main objective is to train a Recurrent Neural Network (RNN), composed of convolutional layers too, in order to catch 
short-term and long-term patterns in the game's board (positioning of the tokens).\
\
The main issue of this project is the dataset: to train a RNN (or any other model) we need to teach it the play style of a good player, otherwise the 
model will learn sub-optimal game strategies. I then decided to hard-code a simple strategy and make the RNN learn from it. Then, simulating thousands of different games,
I let two RNNs play against each other to generate a large dataset, perhaps containing good patterns and "intelligent" moves, repeating the process multiple times.\
\
The **results** are found in the `main.ipynb` file. For coding details and model architectures, feel free to read (and use) every other file in the repository.
