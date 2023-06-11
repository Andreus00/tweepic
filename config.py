# seed
SEED = 3

# pipeline configuration
n_buckets = 60
n_neighbors = 5
n_words = 3
n_hashtags = 3

# dataset configuration
dataset_path = "data/parquet_dataset"
intermediate_embeddings_path = "intermediate_result"
intermediate_proximity_path = "word_and_hashtags_checkpoint"

# spark configuration

spark_memory = "25g"
executor_memory = "2g"
max_result_memory = "25g"
in_memory_fraction = "0.8"