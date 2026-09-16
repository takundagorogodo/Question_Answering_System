"""
Create 11-53 corpus skeletons with outlines.
Run: python create_corpus_expansion.py
Then HUMAN-REVIEW each file, expand to 400-800 chars, original wording.
After: python -m src.preprocessing; python -m src.chunking; python -m src.vector_store; python -m src.evaluate
"""
from pathlib import Path

docs = {
"11_stemming.txt": "Stemming: definition, crude suffix chopping, Porter/Snowball stemmer example (running->run), fast but errors, vs lemmatization.",
"12_stop_words_and_text_normalization.txt": "Stop words (is, the, and), why remove for TF-IDF but keep for transformers, lowercasing, punctuation removal, normalization trade-offs.",
"13_part_of_speech_tagging.txt": "POS tagging: noun/verb/adjective labels, why useful for parsing and NER, rule-based vs statistical vs neural taggers.",
"14_named_entity_recognition.txt": "NER: person/location/org detection, BIO tagging, applications in extraction and QA, challenges with ambiguous names.",
"15_n_grams_and_language_models.txt": "n-grams: unigram/bigram/trigram, counting, smoothing, use in language modeling, limitations vs neural models.",
"16_bag_of_words_and_tf_idf.txt": "BoW: count vectors, TF-IDF weighting TF*IDF, sparse, baseline for classification/retrieval, vs dense embeddings.",
"17_word2vec.txt": "Word2Vec: CBOW and Skip-gram, learns dense vectors from context, king-man+woman=queen example, limitations static embeddings.",
"18_glove.txt": "GloVe: global vectors from co-occurrence matrix, combines count and prediction, similar to Word2Vec but uses global stats.",
"19_fasttext.txt": "FastText: subword n-grams, handles OOV and morphologically rich languages, example: tokenization -> token + iza + tion.",
"20_rnn_lstm_gru.txt": "RNN, LSTM, GRU: sequential models, vanishing gradient, LSTM gates, used for text before transformers.",
"21_sequence_to_sequence_models.txt": "Seq2Seq: encoder-decoder for translation/summarization, bottleneck, attention improves it.",
"22_attention_mechanism.txt": "Attention: weighted focus on input tokens, query/key/value, solves bottleneck, basis for transformers.",
"23_bert.txt": "BERT: bidirectional encoder, masked LM pretraining, fine-tuning for classification/QA, 110M/340M params.",
"24_gpt_and_autoregressive_models.txt": "GPT: decoder-only, autoregressive left-to-right, next token prediction, few-shot prompting.",
"25_text_classification.txt": "Text classification: spam, sentiment, topics, pipeline tokenize->embed->classifier, metrics precision/recall/F1.",
"26_sentiment_analysis.txt": "Sentiment: positive/negative/neutral, lexicon vs ML, challenges sarcasm, domain shift.",
"27_machine_translation.txt": "MT: rule-based -> SMT -> NMT, encoder-decoder, BLEU evaluation, low-resource challenges.",
"28_question_answering.txt": "QA types: extractive (span) vs generative (write answer) vs RAG (retrieve+generate), evaluation.",
"29_text_summarization.txt": "Summarization: extractive (select sentences) vs abstractive (generate), ROUGE metric.",
"30_information_extraction.txt": "IE: structured info from unstructured text, entities + relations + events, pipeline NER+RE.",
"31_dependency_and_constituency_parsing.txt": "Parsing: dependency (head-dependent) vs constituency (phrase structure), treebanks.",
"32_word_sense_disambiguation.txt": "WSD: word has multiple meanings (bank river vs bank money), context decides, Lesk algorithm.",
"33_coreference_resolution.txt": "Coreference: he/she/it refers to earlier entity, example: 'John said he...' he=John, hard for models.",
"34_semantic_analysis.txt": "Semantic: meaning beyond syntax, word sense, semantic roles, NLI entailment.",
"35_topic_modeling.txt": "Topic modeling: LDA, discovers topics from doc collection, bag-of-words, coherence.",
"36_chatbots_and_conversational_ai.txt": "Chatbots: rule-based -> retrieval -> generative, dialogue state, hallucination risk.",
"37_prompt_engineering.txt": "Prompt engineering: instruction, few-shot examples, chain-of-thought, prompt probe A-E at 0.803 score.",
"38_fine_tuning.txt": "Fine-tuning: adapt pretrained model to task with small labeled data, vs prompting, catastrophic forgetting.",
"39_transfer_learning.txt": "Transfer learning: pretrain on large corpus, transfer to downstream, BERT/GPT examples.",
"40_rag_question_answering.txt": "RAG QA: retrieve top-K chunks then generate grounded answer, reduces hallucination vs pure LLM.",
"41_rag_pipeline.txt": "RAG pipeline: chunking 500 chars overlap 100 -> embeddings 384-dim -> FAISS IndexFlatIP -> threshold 0.35 gate -> FLAN-T5-base.",
"42_chunking_and_document_splitting.txt": "Chunking: why split docs (context limit), fixed vs semantic, overlap preserves context, your 60->65 chunks avg 452.",
"43_cosine_similarity.txt": "Cosine similarity: dot product of normalized vectors, 1=identical, 0=unrelated, your 0.521 paraphrase vs 0.033 unrelated.",
"44_precision_recall_f1.txt": "Precision (correct among returned), Recall (found among relevant), F1 harmonic mean, for retrieval/QA.",
"45_bleu_rouge_meteor.txt": "BLEU (MT precision), ROUGE (summarization recall), METEOR (synonyms), all n-gram overlap metrics.",
"46_natural_language_inference.txt": "NLI: premise entails/contradicts/neutral hypothesis, example: 'dog runs' entails 'animal moves'.",
"47_text_classification_algorithms.txt": "Algorithms: Naive Bayes, SVM, Logistic Regression, fine-tuned BERT, trade-offs speed vs accuracy.",
"48_hmm_and_viterbi.txt": "HMM: hidden states, Markov assumption, Viterbi decoding for POS tagging, old but foundational.",
"49_crf.txt": "CRF: conditional random field, discriminative, better than HMM for sequence labeling, uses context features.",
"50_nlp_pipeline.txt": "Typical pipeline: raw text -> tokenization -> normalization -> POS/NER -> embeddings -> model -> output.",
"51_nlp_challenges.txt": "Challenges: ambiguity, sarcasm, low-resource languages, bias, hallucination (your 0.108 OOD).",
"52_nlp_applications.txt": "Applications: search, translation, chatbots, sentiment, QA (your project), summarization.",
"53_nlp_ethics_and_bias.txt": "Ethics: bias in training data, fairness, privacy, hallucination risk, need auditable sources (your differentiator)."
}

corpus_dir = Path("data/documents")
corpus_dir.mkdir(parents=True, exist_ok=True)

for fname, outline in docs.items():
    p = corpus_dir / fname
    if p.exists():
        print(f"Skip exists {fname}")
        continue
    title = fname.replace(".txt","").replace("_"," ").title()
    content = f"{title}\n\n{outline}\n\nExpand this to 2-3 paragraphs, 400-800 chars, original wording, example. Keep it factual and grounded. This file is AI-drafted outline, human must review and expand with disclosure in SOURCES.md.\n"
    p.write_text(content, encoding="utf-8")
    print(f"Created {fname}")

print("\nDone. Now HUMAN-REVIEW each file, expand to full paragraphs, then run:")
print("python -m src.preprocessing; python -m src.chunking; python -m src.vector_store; python -m src.evaluate")
