from wordvectors import FastTextWrapper

# wv stands for word vectors.
# FastTextWrapper takes one arg: the directory of your data
wv = FastTextWrapper("../data")

# Training will take a pretty long time. Mine took about 25 min.
wv.train_model("../data/wv_train.txt", vector_len=100)

# Only need to train once. You can load the next time.
# Model is saved in ../data/fasttext
# wv.load_model()

# Creates a pickle obj that is a dictionary with keys as words
# and values as their embeddings
wv.create_embeddings_dict()

# Returns the vector embedded version of your input string
print(wv.embedding_lookup("This is a test string"))

# Returns the nearest neighbours to the input and their euclidean distance
wv.find_nn("truck", num_neighbors=10)
