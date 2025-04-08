def rank_of_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    rank = 0

    for r in range(min(rows, cols)):
        # Step 1: Find a pivot
        if matrix[r][r] == 0:
            for i in range(r+1, rows):
                if matrix[i][r] != 0:
                    matrix[r], matrix[i] = matrix[i], matrix[r]  # Swap
                    break
        if matrix[r][r] == 0:
            continue

        # Step 2: Eliminate below the pivot
        for i in range(r+1, rows):
            ratio = matrix[i][r] / matrix[r][r]
            for j in range(cols):
                matrix[i][j] -= ratio * matrix[r][j]

    # Step 3: Count non-zero rows
    for row in matrix:
        if any(abs(val) > 1e-10 for val in row):  # account for float precision
            rank += 1

    return rank


matrix1 = [
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
]

print(rank_of_matrix(matrix1))


