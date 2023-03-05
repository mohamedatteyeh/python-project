import enum
import numpy as np


class GridPosition(enum.Enum):
    empty = 0
    yellow = 1
    red = 2

class Grid:
    def __init__(self,rows,columns):
        self._rows = rows
        self._columns = columns
        self._grid = None
        self.initGrid()

    def initGrid(self):
        self._grid = np.zeros((self._columns,self._rows))
        
    
    def getGrid(self):
        return self._grid

    def PlacePiece(self,column,piece):
        if column < 0 | column > self._columns:
            raise ValueError ('invalid column')
        if piece not in [i for i in range(3)]:
            raise ValueError ('Invalid piece')
        for row in range (self._rows-1,-1,-1):
            if self._grid[row][column] == 0:
                self._grid[row][column] = piece # Updating the positional value 
    def win(self,connectnum,row,col,piece):
        count_row = 0
        count_column = 0
        count_diagonal = 0
        count_antidiagonal = 0
        
        #checking verticality
        for r in range(self._rows):
            if self._grid [r][col] == piece:
                count_row += 1
            else: 
                count_row = 0
        
        #Checking Horizantaly 
        

def main():
    a = Grid(3,3).PlacePiece(2,1)
    print(a)

if __name__ == "__main__":
    main()