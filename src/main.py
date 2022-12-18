from preprocess import *
from train import *
from utils import *
from topic2vec import *
from BertTopic import *
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main(dataset,embeddingmethod,dimReduction,clusterMethod,k,embeddingmodel):
   
    # Loading data and Pre-processing
    Preprocessor = Preprocess(dataset)
    data, labels = Preprocessor.load_data()
    sentences , token_lists, idx_sample_list = Preprocessor.preprocess()
        
    print(" Preprocessing done")
    
    # Train model
    trainer = Trainers(embeddingmethod=embeddingmethod,dimReduction=dimReduction,clusterMethod=clusterMethod,k=k,data = sentences,token_lists=token_lists,embeddingmodel=embeddingmodel)
   
    if embeddingmethod not in ['Bertopic','Top2Vec']:
        tm,model = trainer.train()
    else:
        tm = trainer.train()
    
    print(" Fitting done")
    # Evaluate model
    if embeddingmethod in ["LDA_S_TRANS","LDA_DIFFCSE"]:
        # Get coherence
        print("Model is:",model)
        # coherence_cv , coherence_npmi = get_coherence(tm,model, token_lists)

        # print("-"*50)
        # print("Evaluation results for {}:".format(embeddingmethod))
        # print("-"*50)
        # print("Model       Score")
        # print("-"*50)
        # print("{} Coherence CV: {}".format(embeddingmethod, coherence_cv))
        # print("-"*50)
        # print("{} Coherence NPMI: {}".format(embeddingmethod, coherence_npmi))
        # print("-"*50)
        # print("{} Topic Diversity: {}".format(method, 
        print_evaluations(tm, model, token_lists,labels,k, embeddingmethod)
        # print("Silhoulette:", silhoulette)
        # print("Topics:", topics)
        # return coherence, silhoulette, topics
        
    elif embeddingmethod == "Top2Vec":
        print("Number of topics:", tm.get_num_topics())
        print("Topic words:", tm.get_topics())
        for i in range(tm.get_num_topics()):
        # tm.get_num_topics():
            print("TOP 10 Words for topic:{} --> {}        ",(i,tm.topic_words[i]))
        print("topic words:", tm.topic_words)
        evaluate_top2vec(tm.topic_words,sentences)
        for topic in range(tm.get_num_topics()):
            tm.generate_topic_wordcloud(topic,background_color='white')
        
    elif embeddingmethod == "Bertopic":
        topic_list = []
        print("Total number of topics are:",len(tm.get_topics()))
        for i in range(len(tm.get_topics())-1):
            for j in tm.get_topic(topic=i):
                topic_list.append(j[0])
        print("topic_list:",topic_list)
        evaluate_bertopic(topic_list,sentences) #TODO: add clustering method
        
        # Add bar chart for each topic
        tm.visualize_barchart()



if __name__ == '__main__':
    
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='short', help='Dataset to use')
    parser.add_argument('--embeddingmethod', type=str, default = 'LDA_S_TRANS', help='Embedding method to use')
    parser.add_argument('--dimReduction', type=str, default = 'UMAP', help='Dimensionality reduction method to use')
    parser.add_argument('--clusterMethod', type=str, default = 'KMeans', help='Clustering method to use')
    parser.add_argument('--k', type=int, default = 5, help='Number of clusters')
    parser.add_argument('--embeddingmodel', type=str, default = "all-MiniLM-L6-v2", help='Embedding model to use')
    args = parser.parse_args()
    
    # self.embedding_models = ["all-distilroberta-v1","all-MiniLM-L6-v2","multi-qa-MiniLM-L6-cos-v1","paraphrase-MiniLM-L3-v2"]

    main(dataset='medium',embeddingmethod='LDA_DIFFCSE',dimReduction="UMAP",clusterMethod="agglomerative",k=10,embeddingmodel="paraphrase-MiniLM-L3-v2")