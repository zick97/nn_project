import numpy as np
import pandas as pd
import tensorflow as tf
#---------------------------------------------------------------------------------------------
class Player(object):
    def __init__(self, p=1, name='Player1'):
        self.player = p                     # number ID to produce the grid markers
        self.name = name                    # the name is used only for displaying purposes and to avoid misunderstandings
        if self.player == 1:
            self.marker = 1
        else:
            self.marker = -1
        self.target = int(4 * self.marker)  # +4 for player 1 and -4 for player 2

#---------------------------------------------------------------------------------------------       
class HumanPlayer(Player):
    def __init__(self, p=1, name='Human'):
        Player.__init__(self, p, name)
        self.player_type = 'Human'

    def move(self, Board):
        self.choice = int(input('Digit a column number to place your token.'))
        
class RandomPlayer(Player):
    def __init__(self, p=1, name='Random'):
        Player.__init__(self, p, name)
        self.player_type = 'RandomAI'

    def move(self, Board):
        # columns with available moves
        valid_columns = [i for i, v in enumerate(Board.column_n_pos) if v != 0]
        self.choice = np.random.choice(valid_columns)

class SimplePlayer(Player):
    def __init__(self, p=1, name='Simple'):
        Player.__init__(self, p, name)
        self.player_type = 'SimpleAI'

    def move(self, Board):
        # columns with available moves
        valid_columns = [i for i, v in enumerate(Board.column_n_pos) if v != 0]
        # first of all, it guesses at random 
        # then, using the following algorithm, it checks for rows, columns or diagonals that
        # may lead to victory or defeat and fills the column where there is still a 0
        rows_number = (Board.width - 3) * Board.height  # number of checking rows
        cols_number = Board.width * (Board.height - 3)  # number of checking columns
        # below we write, respectively, the checking rows, columns and diagonals vectors stored
        # inside the board class
        rows = Board.vectors[:rows_number]
        cols = Board.vectors[rows_number:(rows_number + cols_number)]
        diags = Board.vectors[(rows_number + cols_number):]
        # now we want to identify the column of each element inside these vectors
        # to do so, we use an equation explained in the markdown file
        if self.target == 4:
            target = 3
        else:
            target = -3

        column = -1 # initialize the column to a non-valid value
        for i, vector in enumerate([rows, cols, diags]):
            targets = np.array([sum(v) for v in vector])           # sum of each vector (row, col, or diag) 
            vector_index = np.where(targets == -(target))          # list of indexes where the vector can be completed
            if len(vector_index[0]) > 0:
                index = np.where(vector[vector_index[0][0]] == 0)  # index of the 0 inside the vector
                if i == 0:
                    column = int(vector_index[0][0] / Board.height) + index[0][0]
                if i == 1:
                    column = vector_index[0][0] % Board.width
                if i == 2:
                    column = int(int(vector_index[0][0] / 2) / (Board.height - 3)) + index[0][0]
                break
            # the same thing can happen for a winning move: by putting it after the defeat scenario,
            # we give priority to winning moves
            vector_index = np.where(targets == target)
            if len(vector_index[0]) > 0:
                index = np.where(vector[vector_index[0][0]] == 0)
                if i == 0:
                    column = int(vector_index[0][0] / Board.height) + index[0][0]
                if i == 1:
                    column = vector_index[0][0] % Board.width
                if i == 2:
                    column = int(int(vector_index[0][0] / 2) / (Board.height - 3)) + index[0][0]
                break

        if column in valid_columns:
            self.choice = column
        else:
            self.choice = np.random.choice(valid_columns)