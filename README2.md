What is Mutual Information?

    Mutual Information (MI) measures how much information one variable tells us about another.

It answers:

    "If I know the value of X, how much does that reduce my uncertainty about Y?"

Score individual features based on how much info they give about the target.

- its Supervised (uses target info!)
- Captures both linear & nonlinear relationships

- MI score tells how useful each feature is You keep the original features. but when we use PCA it gives Transformed "components" (not original features)

- PCA only captures Only linear (unless you use Kernel PCA) and      Unsupervised (ignores target). but when we use MI, it is Supervised  (uses target info!). 

- MI Score individual features based on how much info they give about the target

# (SVD) Single value decomposition

In Simple Words:

    SVD breaks down a matrix into 3 smaller, special matrices — kind of like breaking a complex shape into its core building blocks.

## What is SVD?

Let’s say you have a real matrix A (could be any shape — not necessarily square).
Then SVD says:

- A=U⋅Σ⋅VT

Where:

    AA is an m×nm×n matrix

    UU is an m×mm×m orthogonal matrix (columns are unit vectors, like basis for input space)

    ΣΣ is an m×nm×n diagonal matrix with singular values (non-negative real numbers, sorted in descending order)

    VTVT is the transpose of an n×nn×n orthogonal matrix (basis for the output space)


 Step-by-Step Plan

    Matrix input

    Transpose

    Matrix multiplication

    Characteristic polynomial

    Solve eigenvalues using determinant

    Find eigenvectors

    Normalize vectors

    Build U, Σ, V

    Show result    

U: Left singular vectors
Σ: Diagonal matrix of singular values
VT: Right singular vectors (transposed)    