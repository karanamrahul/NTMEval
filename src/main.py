from preprocess import *
from train import *
from lda_bert import *
from utils import *
from topic2vec import *
from bertTopic import *

# from lda_umap import *
# from HCluster import *
# from agglomerative import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(dataset,method,k):
   
    # Loading data and Pre-processing
    Preprocessor = Preprocess(dataset)
    data = Preprocessor.load_data()
    sentences , token_lists, idx_sample_list = Preprocessor.preprocess()
        
    print(" Preprocessing done")
    
    # Train model
    trainer = Trainer(method=method,k=k,data = sentences,token_lists=token_lists)
    model = trainer.train()
    print(" Training done")
    
    # # Fit model
    # model.fit(sentences,token_lists)
    
    print(" Fitting done")
    # Evaluate model
    
    if method == "LDA_BERT" or method == "SROBERTA":
        # Get coherence
        coherence_cv , coherence_npmi = get_coherence(model, token_lists)
        # diversity = get_topic_diversity(model, topk_words)
        # # Get silhoulette
        # silhoulette = get_silhoulette(model)
        # # Plot UMAP
        # plot_project(model.vec[model.method], model.cluster_model.labels_)
        # # Get topics
        # topics = get_topic_words(token_lists, model.cluster_model.labels_)
        # # Print results
        # print("Token lists:", token_lists[1:10])
        print("-"*50)
        print("Evaluation results for {}:".format(method))
        print("-"*50)
        print("Model       Score")
        print("-"*50)
        print("{} Coherence CV: {}".format(method, coherence_cv))
        print("-"*50)
        print("{} Coherence NPMI: {}".format(method, coherence_npmi))
        print("-"*50)
        # print("{} Topic Diversity: {}".format(method, 
        
        # print("Silhoulette:", silhoulette)
        # print("Topics:", topics)
        # return coherence, silhoulette, topics
        
    elif method == "Top2Vec":
        print("Number of topics:", model.get_num_topics())
        print("Topic words:", model.get_topics())
        for i in range(model.get_num_topics()):
        # model.get_num_topics():
            print("TOP 10 Words for topic:{} --> {}        ",(i,model.topic_words[i]))
        evaluate_top2vec(model.topic_words,sentences)
        for topic in range(model.get_num_topics()):
            model.generate_topic_wordcloud(topic,background_color='white')
        
    elif method == "Bertopic":
        topic_list = []
        print("Total number of topics are:",len(model.get_topics()))
        for i in range(len(model.get_topics())-1):
            for j in model.get_topic(topic=i):
                topic_list.append(j[0])
        print("topic_list:",topic_list)
        evaluate_bertopic(topic_list,sentences)
        
        # Add bar chart for each topic
        model.visualize_barchart()

if __name__ == '__main__':
    
    
    # Test out the possible pa
    main(dataset='short',method='Top2Vec',k=5)