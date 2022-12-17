from DiffCSE.diffcse import DiffCSE
from preprocess import *
model = DiffCSE("voidism/diffcse-bert-base-uncased-sts")

embeddings = model.encode("A woman is reading.")

print("Embeddings shape:",embeddings.shape)

dataset = "20newsgroup"
Preprocessor = Preprocess(dataset)
data = Preprocessor.load_data()
sentences , token_lists, idx_sample_list = Preprocessor.preprocess()


