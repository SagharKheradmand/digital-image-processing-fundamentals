def read_matrix(name):
    print(f"Enter dimensions for matrix {name}:")
    rows = int(input("  Number of rows: "))
    cols = int(input("  Number of columns: "))

    print(f"Enter matrix {name} row by row (space-separated numbers):")
    matrix = []
    for i in range(rows):
        row = list(map(int, input(f"  Row {i+1}: ").split()))
        if len(row) != cols:
            raise ValueError("Incorrect number of elements in row.")
        matrix.append(row)
    return matrix


# Read S (small matrix)
S = read_matrix("S")   # size n × m
n = len(S)
m = len(S[0])

# Read L (large matrix)
L = read_matrix("L")   # size H × W
H = len(L)
W = len(L[0])

# Compute center of L
cx = H // 2
cy = W // 2

# Diamond rotation formula
# row = cx + (i - j)
# col = cy + (i + j - (m - 1))

for i in range(n):
    for j in range(m):
        row = cx + (i - j)
        col = cy + (i + j - (m - 1))

        # Check bounds before writing
        if not (0 <= row < H and 0 <= col < W):
            print("Impossible")
            exit()

        L[row][col] = S[i][j]

# Output result
print("\nResulting matrix L:")
for row in L:
    print(*row)
