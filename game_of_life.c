/*
 * Name: Brendan Caudill
 * ID: caudil77
 * Homework#: 2
 * To compile: gcc -fopenmp -o gameOfLife game_of_life.c
 * To run: ./gameOfLife <board size> <generations> <threads>
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

// Define constants for alive and dead cells
#define ALIVE 1
#define DEAD 0


/**
 * Updates the ghost cells of a 2D board for periodic boundary conditions.
 * Ghost cells are used to simulate a toroidal (donut-shaped) topology where
 * the edges of the board wrap around to the opposite side, making the
 * simulation space continuous. This function copies the edge cells to their
 * corresponding opposite edges to implement this behavior.
 *
 * @param board A pointer to a pointer of integers representing the game board,
 *              where board is dynamically allocated with size (N+2) x (N+2) to
 *              include ghost cells on all four edges.
 * @param N The size of the actual game board, excluding the ghost cells. The
 *          total size of the board including ghost cells is (N+2) x (N+2).
 */
void ghostCells(int **board, int N){
         // Copying bottom cells to top ghost cells
    for (int i = 0; i <= N+1; i++) {
        board[0][i] = board[N][i];
    }

    // Copying top cells to bottom ghost cells
    for (int i = 0; i <= N+1; i++) {
        board[N+1][i] = board[1][i];
    }

    // Copying right cells to left ghost cells
    for (int i = 0; i <= N+1; i++) {
        board[i][0] = board[i][N];
    }

    // Copying left cells to right ghost cells
    for (int i = 0; i <= N+1; i++) {
        board[i][N+1] = board[i][1];
    }

}

/**
 * Initializes the board for a cellular automaton (e.g., Game of Life) with a random state.
 * Each cell within the actual game area (excluding ghost cells) is randomly set to either
 * alive (1) or dead (0). After initializing the real cells, the ghost cells are updated to
 * maintain periodic boundary conditions. This setup provides an initial random configuration
 * for the simulation to evolve from.
 * 
 * @param board A pointer to a pointer of integers representing the game board. The board
 *              must be dynamically allocated with size (N+2) x (N+2) to include a border
 *              of ghost cells around the actual game area. These ghost cells facilitate
 *              the implementation of periodic boundary conditions.
 * @param N     The size of the actual game area. The board's total size including ghost
 *              cells is (N+2) x (N+2), where the additional rows and columns represent
 *              the ghost cells at the edges.
 */
void initializeBoard(int **board, int N) {
    // Populate the board with random states. Loop through each cell within
    // the actual game area, excluding the ghost cell borders. For each cell,
    // assign a random state: ALIVE (1) or DEAD (0).
    unsigned int seeds[omp_get_max_threads()];
    for(int i = 0; i < omp_get_max_threads(); i++) {
	    seeds[i] = time(NULL) ^ i;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
	    int thread_id = omp_get_thread_num();
            board[i][j] = rand_r(&seeds[thread_id]) % 2;
        }
    }
	
    // After initializing the real cells, update the ghost cells to reflect
    // the periodic boundary conditions. This step ensures that the simulation's
    // edges wrap around, creating a continuous, toroidal space.
    ghostCells(board, N);
}

/**
 * Retrieves the current time with microsecond precision.
 *
 * This function uses the `gettimeofday` function to obtain the current time,
 * then combines the seconds and microseconds components into a single double
 * value representing the total time in seconds. This high-resolution time
 * measurement is useful for performance benchmarking or time-based calculations
 * in simulations and other time-sensitive applications.
 *
 * @return A double value representing the current time in seconds with
 *         microsecond precision.
 */

double gettime(void) {
	struct timeval tval; // Structure to store the current time
	gettimeofday(&tval, NULL); // Retrieve the current time
    // Convert the time to double, combining seconds and microseconds
	return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

/**
 * Prints the contents of a 2D integer array to the standard output.
 * 
 * This function iterates over each row and column of the given 2D array and
 * prints the elements in a formatted grid layout. Each element is followed by
 * a space, and each row is terminated with a newline character. This is useful
 * for visualizing the state of arrays in console-based applications, such as
 * displaying the board state in a game or the output of a matrix operation.
 * 
 * @param a     A pointer to a pointer of integers representing the 2D array to
 *              be printed. The array is assumed to be dynamically allocated.
 * @param mrows The number of rows in the array.
 * @param ncols The number of columns in the array.
 */

void printarray(int **a, int mrows, int ncols) {
	int i,j; // Loop variables
    // Iterate over each row
	for (i=0; i<mrows; i++) {
	// Iterate over each column in the current row
	   for (j=0; j<ncols; j++)
		printf("%d ", a[i][j] % 2); // Print the current element
		printf("\n"); // Newline after each row for a grid layout
	}
}

/**
 * Counts the number of alive neighbors around a specific cell in a 2D board.
 * 
 * This function calculates the number of alive neighbors for a given cell
 * located at coordinates (x, y) on the board. It considers the eight
 * surrounding cells (diagonals, verticals, and horizontals) and counts how
 * many of them are in the ALIVE state. The function ensures not to count
 * the cell itself. This count is crucial for applying the rules of cellular
 * automata like Conway's Game of Life, where the next state of a cell depends
 * on the number of its alive neighbors.
 * 
 * @param board A pointer to a pointer of integers representing the 2D board.
 *              The board is assumed to be dynamically allocated and includes
 *              an additional border (ghost cells) around the actual game area
 *              to simplify boundary conditions.
 * @param x     The x-coordinate (row) of the cell for which to count alive
 *              neighbors. This coordinate is based on the actual game area,
 *              excluding any ghost cell borders.
 * @param y     The y-coordinate (column) of the cell for which to count alive
 *              neighbors. This coordinate is also based on the actual game area,
 *              excluding ghost cells.
 * @param N     The size of the board, not including the border of ghost cells.
 *              This parameter is used to prevent accessing out-of-bound areas
 *              when calculating neighbors near the edges.
 * 
 * @return The number of alive neighbors around the cell at (x, y).
 */

int countAliveNeighbors(int **board, int x, int y, int N) {
    int count = 0; // Initialize the count of alive neighbors to 0

    // Loop over each of the eight surrounding cells
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) {
		// Skip the cell itself
                continue;
            }

	    // Check if the neighbor is alive; note that the board's edges wrap
            // around, so we must consider boundary conditions separately
            if (board[x + dx][y + dy] == ALIVE) {
                count++; // Increment count if the neighbor is alive
            }
        }
    }

    return count; // Return the total count of alive neighbors
}

/**
 * Evolves the state of the board to the next generation according to the rules
 * of the Game of Life. This function creates a
 * new board to calculate the next state of each cell based on the current state
 * and the number of alive neighbors it has. Once the new state for all cells is
 * determined, the function updates the original board with these new states.
 * 
 * @param board A double pointer to an integer array representing the game board,
 *              dynamically allocated with size (N+2) x (N+2) to include a border
 *              of ghost cells around the actual game area, facilitating edge
 *              case handling.
 * @param N     The size of the actual game area, not including the ghost cell
 *              border. The total size of the board, including ghost cells, is
 *              therefore (N+2) x (N+2).
 */
void evolveBoard(int **oldBoard, int **board, int N) {

	#pragma omp parallel for collapse(2)
	for(int x = 1; x <= N; x++) {
		for (int y = 1; y <= N; y++) {
			int aliveNeighbors = countAliveNeighbors(oldBoard, x, y, N);
			if (oldBoard[x][y] = ALIVE) {
				board[x][y] = (aliveNeighbors == 2 || aliveNeighbors == 3) ? ALIVE : DEAD;
			}
			else {
				board[x][y] = (aliveNeighbors == 3) ? ALIVE : DEAD;
			}
		}
	}
	ghostCells(board, N);
}

/**
 * The main entry point for the Game of Life simulation program.
 *
 * This program simulates the Game of Life on a board of size N x N over a
 * specified number of generations, or until the board state stops changing.
 * The initial state of the board is randomized. The program measures and
 * outputs the execution time for the simulation.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments. Expects two arguments:
 *             - N: The size of the board (N x N).
 *             - maxGenerations: The maximum number of generations to simulate.
 *
 * @return Returns 0 on successful completion.
 */
int main(int argc, char *argv[]) {
    // Variables for the board size, number of generations, and timing
    int N; // Size of the board
    int maxGenerations; // Maximum number of generations
    int numThreads;
    double starttime, endtime;

    // Seed the random number generator for board initialization
    srand(time(NULL));

    // Validate command-line arguments
    if (argc != 4) {
	printf("Usage: %s <N> <maxGenerations> <numThreads>\n", argv[0]);
	exit(-1);
    }
	
    // Parse board size and maximum generations from command-line arguments
    N = atoi(argv[1]);
    maxGenerations = atoi(argv[2]);
    numThreads = atoi(argv[3]);
    starttime = gettime();
    omp_set_num_threads(numThreads);

    // Dynamically allocate memory for the board and a copy of it (oldBoard)
    int **board = (int **)malloc((N + 2) * sizeof(int *));
    int **oldBoard = (int **)malloc((N+2) * sizeof(int *));
    for (int i = 0; i < N + 2; i++) {
        board[i] = (int *)malloc((N + 2) * sizeof(int));
	oldBoard[i] = (int *)malloc((N+2) * sizeof(int));
    }
    
    // Initialize the board with a random state
    initializeBoard(board, N);

    // Main loop to evolve the board through generations
    for (int generation = 1; generation <= maxGenerations; generation++) {

        // Evolve the board to the next generation
	evolveBoard(oldBoard, board, N);

        // Print the current state of the board
        // printarray(board, N, N);
	// printf("\n");

	 int **temp = oldBoard;
	 oldBoard = board;
	 board = temp;
	
    }

    // Measure and print the execution time
    endtime = gettime();
    printf("Time taken = %lf seconds\n", endtime-starttime);

    free(board);
    free(oldBoard);


    return 0;
}

