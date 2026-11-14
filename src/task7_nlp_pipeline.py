"""Task 7: Integrated NLP Pipeline."""
from collections import Counter
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task5_word2vec import train_word2vec
from src.task6_Ngrams import generate_ngrams


def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]:
    """
    Perform comprehensive NLP analysis on a collection of documents.

    This pipeline integrates text preprocessing, vectorization, word embeddings,
    and n-gram analysis to provide a complete document analysis.

    Args:
        documents (List[str]): List of text documents to analyze.
        use_lemmatization (bool): If True, use lemmatization; else use stemming.
                                  Default is True.
        ngram_range (Tuple[int, int]): Range of n-gram sizes for vectorization.
                                        Default is (1, 2).
        vector_method (str): Vectorization method - "bow" or "tfidf".
                            Default is "tfidf".

    Returns:
        Dict[str, Any]: Comprehensive analysis results including preprocessed
                       documents, vectorization, embeddings, and n-gram analysis.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty, has fewer than 2 documents,
                   vector_method is invalid, or ngram_range is invalid.

    Example:
        >>> docs = ["I love NLP", "NLP is fun", "Machine learning rocks"]
        >>> results = analyze_documents(docs)
        >>> results['preprocessed']['total_tokens']
        9
    """
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if not documents:
        raise ValueError("Documents cannot be empty")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")
    if any(not isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")
    if vector_method not in {"bow", "tfidf"}:
        raise ValueError("vector_method must be 'bow' or 'tfidf'")
    if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
        raise ValueError("Invalid ngram_range")
    if any(not isinstance(value, int) or value < 1 for value in ngram_range):
        raise ValueError("Invalid ngram_range")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError("Invalid ngram_range")

    preprocessed_docs: List[List[str]] = []
    all_tokens: List[str] = []

    for doc in documents:
        tokens = clean_and_tokenize(doc)
        processed = stem_and_lemmatize(tokens)
        normalized_tokens = processed["lemmatized"] if use_lemmatization else processed["stemmed"]
        preprocessed_docs.append(normalized_tokens)
        all_tokens.extend(normalized_tokens)

    total_tokens = sum(len(doc_tokens) for doc_tokens in preprocessed_docs)
    unique_tokens = len(set(all_tokens))
    avg_doc_length = total_tokens / len(documents)

    vectorized_texts = [" ".join(doc_tokens) for doc_tokens in preprocessed_docs]
    if vector_method == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    feature_matrix = vectorizer.fit_transform(vectorized_texts)
    feature_df = pd.DataFrame(feature_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    if preprocessed_docs and all(preprocessed_docs):
        sentence_corpus = [doc_tokens for doc_tokens in preprocessed_docs if doc_tokens]
        word2vec_model = train_word2vec(sentence_corpus, vector_size=100, window=5, min_count=1)
        vocabulary_size = len(word2vec_model.wv)
        vector_size = word2vec_model.wv.vector_size
    else:
        word2vec_model = None
        vocabulary_size = 0
        vector_size = 0

    ngram_items: List[Tuple[str, ...]] = []
    for doc_tokens in preprocessed_docs:
        doc_text = " ".join(doc_tokens)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if doc_text:
                ngram_items.extend(generate_ngrams(doc_text, n))

    bigram_counts = Counter(ngram for ngram in ngram_items if len(ngram) == 2)
    trigram_counts = Counter(ngram for ngram in ngram_items if len(ngram) == 3)
    top_bigrams = sorted(bigram_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    top_trigrams = sorted(trigram_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    bigram_diversity = len(bigram_counts) / max(1, len(bigram_counts) + len(trigram_counts))

    similarity_matrix = cosine_similarity(feature_df.to_numpy())
    similarity_df = pd.DataFrame(similarity_matrix, index=range(len(documents)), columns=range(len(documents)))
    similarity_df = similarity_df.round(4)
    for idx in range(len(similarity_df)):
        similarity_df.iloc[idx, idx] = 1.0

    results = {
        "preprocessed": {
            "documents": preprocessed_docs,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": round(avg_doc_length, 2)
        },
        "vectorization": {
            "method": vector_method,
            "feature_matrix": feature_df,
            "feature_names": list(feature_df.columns),
            "shape": feature_df.shape
        },
        "word_embeddings": {
            "model": word2vec_model,
            "vocabulary_size": vocabulary_size,
            "vector_size": vector_size
        },
        "ngram_analysis": {
            "top_bigrams": list(top_bigrams),
            "top_trigrams": list(top_trigrams),
            "bigram_diversity": round(float(bigram_diversity), 3)
        },
        "document_similarity": similarity_df
    }

    return results


def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the most similar documents to a given document.

    Args:
        doc_index (int): Index of the document to find similarities for.
        similarity_matrix (pd.DataFrame): Document similarity matrix.
        top_n (int): Number of similar documents to return. Default is 3.

    Returns:
        List[Tuple[int, float]]: List of (document_index, similarity_score) tuples,
                                 sorted by similarity in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If similarity_matrix is not a DataFrame.

    Example:
        >>> sim_matrix = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]])
        >>> find_similar_documents(0, sim_matrix, top_n=1)
        [(1, 0.8)]
    """
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int):
        raise ValueError("doc_index must be between 0 and len(similarity_matrix)-1")
    if doc_index < 0 or doc_index >= len(similarity_matrix):
        raise ValueError("doc_index must be between 0 and len(similarity_matrix)-1")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    scores = similarity_matrix.iloc[doc_index].copy()
    scores = scores.drop(labels=doc_index, errors='ignore')
    sorted_scores = scores.sort_values(ascending=False)[:top_n]
    return [(int(index), round(float(score), 4)) for index, score in sorted_scores.items()]


def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Extract top keywords from a document based on feature scores.

    Args:
        doc_index (int): Index of the document to extract keywords from.
        feature_matrix (pd.DataFrame): Feature matrix (BoW or TF-IDF).
        top_n (int): Number of top keywords to return. Default is 5.

    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples,
                                 sorted by score in descending order.

    Raises:
        ValueError: If doc_index is out of range or top_n is invalid.
        TypeError: If feature_matrix is not a DataFrame.

    Example:
        >>> features = pd.DataFrame({'nlp': [0.9, 0.2], 'fun': [0.5, 0.8]})
        >>> get_document_keywords(0, features, top_n=2)
        [('nlp', 0.9), ('fun', 0.5)]
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int):
        raise ValueError("doc_index must be between 0 and len(feature_matrix)-1")
    if doc_index < 0 or doc_index >= len(feature_matrix):
        raise ValueError("doc_index must be between 0 and len(feature_matrix)-1")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    row = feature_matrix.iloc[doc_index].sort_values(ascending=False)
    top_features = row.head(top_n)
    return [(str(feature), round(float(score), 4)) for feature, score in top_features.items()]


def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Compare BoW and TF-IDF representations side-by-side.

    Args:
        documents (List[str]): List of text documents to compare.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'bow' and 'tfidf' DataFrames.

    Raises:
        TypeError: If documents is not a list or contains non-strings.
        ValueError: If documents is empty or has fewer than 2 documents.

    Example:
        >>> docs = ["I love NLP", "NLP is fun"]
        >>> comparison = compare_vectorization_methods(docs)
        >>> 'bow' in comparison and 'tfidf' in comparison
        True
    """
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if not documents:
        raise ValueError("Documents cannot be empty")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")
    if any(not isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")

    bow_results = analyze_documents(documents, vector_method="bow")
    tfidf_results = analyze_documents(documents, vector_method="tfidf")

    return {
        "bow": bow_results["vectorization"]["feature_matrix"],
        "tfidf": tfidf_results["vectorization"]["feature_matrix"]
    }


if __name__ == "__main__":
    # Example usage
    sample_documents = [
        "Natural language processing is amazing and fun to learn.",
        "Machine learning and NLP are related fields in AI.",
        "Deep learning models are powerful for NLP tasks.",
        "Python is great for machine learning and data science."
    ]
    
    print("=" * 80)
    print("INTEGRATED NLP PIPELINE ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = analyze_documents(
        sample_documents,
        use_lemmatization=True,
        ngram_range=(1, 2),
        vector_method="tfidf"
    )
    
    # Display results
    print("\n1. PREPROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total tokens: {results['preprocessed']['total_tokens']}")
    print(f"Unique tokens: {results['preprocessed']['unique_tokens']}")
    print(f"Average document length: {results['preprocessed']['avg_doc_length']}")
    
    print("\n2. VECTORIZATION")
    print("-" * 80)
    print(f"Method: {results['vectorization']['method'].upper()}")
    print(f"Feature matrix shape: {results['vectorization']['shape']}")
    print(f"Number of features: {len(results['vectorization']['feature_names'])}")
    
    print("\n3. WORD EMBEDDINGS")
    print("-" * 80)
    print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
    print(f"Vector dimensionality: {results['word_embeddings']['vector_size']}")
    
    print("\n4. N-GRAM ANALYSIS")
    print("-" * 80)
    print(f"Top 5 bigrams:")
    for bigram, count in results['ngram_analysis']['top_bigrams'][:5]:
        print(f"  {bigram}: {count}")
    print(f"Bigram diversity: {results['ngram_analysis']['bigram_diversity']}")
    
    print("\n5. DOCUMENT SIMILARITY")
    print("-" * 80)
    print("Similarity matrix:")
    print(results['document_similarity'])
    
    print("\n6. SIMILAR DOCUMENTS TO DOC_0")
    print("-" * 80)
    similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
    for doc_idx, similarity in similar_docs:
        print(f"  Document {doc_idx}: {similarity:.4f}")
    
    print("\n7. TOP KEYWORDS IN DOC_0")
    print("-" * 80)
    keywords = get_document_keywords(0, results['vectorization']['feature_matrix'], top_n=5)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
    
    print("\n8. COMPARISON: BOW vs TF-IDF")
    print("-" * 80)
    comparison = compare_vectorization_methods(sample_documents[:3])
    print("BoW shape:", comparison['bow'].shape)
    print("TF-IDF shape:", comparison['tfidf'].shape)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
