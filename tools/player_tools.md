The main issue about the `SimplePlayer()` class is that, when he recognizes any row, column or diagonal that is almost full (e.g.: it sums up to 3) it chooses the column to fill the empty space without worrying if this move would end in a defeat. For example, if there is an almost full line at row 3, from column 0 to 3, it starts putting its tokens in column 4 until the empty space gets filled by any token.\
This is basically a serious lack of foresight skills and, although it can be fixed with a bit of hard-coding, I decided to let the real Artificial Intelligence learn this skill with a huge dataset.