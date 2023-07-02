### Game Tools
In this file you can find the whole building process of the game *Connect4*, from the implementation of the basic rules to the function that instantiates and simulates multiple games. You can also find the `HumanPlayer()` class, that gives the user the opportunity to play the game, as well as the basic hard-coded player class `RandomPlayer()`, which plays a random game strategy.\
\
\
__*Player class*__\
First of all, we need to create the `Player()` class: this is a parent class for all the future *player* objects, provinding the essential information required to update the board state, to distinguish the two players and to check who has won the game.\
In fact, the class only contains the *constructor* function.\
Every sub-class will inherit the previous features, as well as an additional function that determines the *child* player's next move.\
\
__*Board class*__\
Then, we need to create a `Board()` class. Its main functions are displaying the grid for the game, finding all the possible valid positions for the tokens, keeping track of the tokens positioning during the game, checking for a winner.\
Notice that the player's move (from the *choice* variable derived from `Player.choice` in the code cell below) is just a single integer value: it represents the column where the player is placing the token.\
\
\
__*First Player*: Human__\
The `HumanPlayer()` move is decided by the user itself.\
\
__*Second Player*: Random__\
The first player is a newbie and does not really know how to play. Therefore, it is playing randomly.\
The only constraints are provided by the conditions written inside the `Board()` class.\
\
\
__*Game engine*__\
Now we need to create the object able to instantiate all the other objects (`Board()` and `Player()`'s child classes) in order to get a full game of *Connect4*, being it simulated or actually played by the users.\
Note that this class requires a `game_type` string argument, defined as follows:
1. *`user-user`*: classic game 1v1, played by human users to overcome the boredom of the NNs' training process,
2. *`random-random`*: random player vs random player,
3. *`user-random`*: user versus random player (used mainly for debugging and problem checking),
4. *`user-simple`*: user versus simple hard-coded player,
5. *`simple-random`*: simple hard-coded player versus random player,
6. *`simple-simple`*: two simple hard-coded players playing against each other,
7. *`rnn-simple`*: recurrent network against simple hard-coded player,
8. *`rnn-rnn`*: two recurrent networks against each other (the core of the training process),
9. *`user-rnn`*: the final goal of this project $\rightarrow$ human player against a well-trained neural network.

\
__*Building the dataset*__\
To generate our training datasets, we only have to simulate the games multiple times, saving the results into a bigger `DataFrame()` object.\
During this part of the project, I encountered some troubles with the execution time: as you can see, building the dataset requires the concatenation of two dataframes and, as the loop goes on, the dataset gets bigger and bigger, resulting in a very slow compilation time, along with the inability for the `tqdm` bar to make a significative estimation of the remaining time.\
To speed things up, I decided to split the process into sub-sets of 1000 simulations each, concatenating the results with the bigger dataset only at the end of these subsets, then refreshing the smaller one.\
Despite not being a very elegant solution, it works quite well for this purpose.