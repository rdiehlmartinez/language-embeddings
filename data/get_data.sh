mkdir raw_multilingual_data
wget -c https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus/raw/master/Multilingual_Parallel_Corpus/Multi_lingual_Parallel_corpus_1.zip -O raw_multilingual_data/corpus.zip
unzip raw_multilingual_data/corpus.zip 
mv Multilingual_Parllel_corpus.txt raw_multilingual_data/corpus.txt
rm raw_multilingual_data/corpus.zip
