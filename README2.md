What is Mutual Information?

    Mutual Information (MI) measures how much information one variable tells us about another.

It answers:

    "If I know the value of X, how much does that reduce my uncertainty about Y?"

Score individual features based on how much info they give about the target.

its Supervised (uses target info!)
Captures both linear & nonlinear relationships
MI score tells how useful each feature is You keep the original features. but when we use PCA it gives Transformed "components" (not original features)

PCA only captures Only linear (unless you use Kernel PCA) and Unsupervised (ignores target). but when we use MI, it is Supervised (uses target info!)