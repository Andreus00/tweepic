# seed
SEED = 3

# pipeline configuration
n_buckets = 10
n_neighbors = 12
n_words = 3
n_hashtags = 9

NUM_FEATURES_CLASSIFICATION = 3

# dataset and checkpoint configuration
dataset_path = "data/parquet_dataset"

intermediate_embeddings_path = "intermediate_result"
intermediate_sentence_proximity_path = "sentence_checkpoint"
intermediate_word_and_hashtags_proximity_path = "word_and_hashtags_checkpoint"
intermediate_edge_classifier_path = "edge_classifier_checkpoint"
intermediate_online_sentence_proximity_path = "online_sentence_checkpoint"

# spark configuration

spark_memory = "20g"
executor_memory = "2g"
max_result_memory = "20g"
in_memory_fraction = "0.6"


classes = [
    '2014-gazaunderattack.ids',
    '2013-boston-marathon-bombing.ids',
    '2014-ebola.ids',
    '2012-uselection.ids',
    '2016-sismoecuador.ids',
    '2012-obama-romney.ids',
    '2014-indyref.ids',
    '2012-sxsw.ids',
    '2015-parisattacks.ids',
    '2016-hijacked-plane-cyprus.ids',
    '2015-refugeeswelcome.ids',
    '2014-stpatricksday.ids',
    '2012-mexican-election.ids',
    '2012-superbowl.ids',
    '2016-panamapapers.ids',
    '2016-irish-ge16.ids',
    '2015-nepalearthquake.ids',
    '2015-hurricanepatricia.ids',
    '2014-ferguson.ids',
    '2014-ottawashooting.ids',
    '2014-hongkong-protests.ids',
    '2014-typhoon-hagupit.ids',
    '2012-hurricane-sandy.ids',
    '2012-euro2012.ids',
    '2015-charliehebdo.ids',
    '2015-germanwings-crash.ids',
    '2016-brexit.ids',
    '2016-brussels-airport-explossion.ids',
    '2016-lahoreblast.ids',
    '2014-sydneysiege.ids',
    'chatgpt'
]


# pipelines
pipeline_model_path = "pipelines/pipeline_model"
sentence_proximity_pipeline_model_path = "pipelines/sentence_proximity_pipeline_model"
word_and_hashtag_proximity_pipeline_model_path = "pipelines/word_and_hashtag_proximity_pipeline_model"
edge_classifier_pipeline_model_path = "pipelines/edge_classifier_pipeline_model"

'''

# neighbors = 8

tot = 38790
p = 1463
n = 37327

tp = 1457
tn = 36631
fn = 6
fp = 696


acc = (tp + tn) / tot = 0.981
prec = tp / (tp + fp) = 0.676
rec = tp / (tp + fn) = 0.996
f1 = 2 * prec * rec / (prec + rec) = 0.805


######

# neighbors = 4

tot = 38790
p = 1463
n = 37327

tp = 1335
tn = 36631
fn = 128
fp = 696

acc = (tp + tn) / tot = 0.976
prec = tp / (tp + fp) = 0.657
rec = tp / (tp + fn) = 0.912
f1 = 2 * prec * rec / (prec + rec) = 0.763

'''