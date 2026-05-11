import streamlit as st
import os
import sys

# Add the parent directory to the Python path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.preprocessing import TextPreprocessor
from src.core.incidence_matrix import IncidenceMatrix
from src.core.inverted_index import InvertedIndex
from src.core.positional_index import PositionalIndex
from src.core.ranking import Ranker
from src.features.spell_checker import SpellChecker
from src.features.query_expansion import QueryExpander
from src.features.snippet_generator import SnippetGenerator

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Mini Search Engine",
    layout="wide"
)

st.title("Mini Search Engine")
st.markdown("A modular search engine demonstrating advanced Information Retrieval concepts.")

# --- 2. Sidebar Settings ---
st.sidebar.header("Settings")
preprocessing_method = st.sidebar.radio(
    "Text Normalization Method:",
    ("lemmatization", "stemming"),
    help="Changing this will rebuild the indices."
)

enable_query_expansion = st.sidebar.checkbox("Enable Query Expansion (WordNet)", value=False)

# --- 3. Caching Engine Initialization ---
@st.cache_resource
def load_and_build_engine(method):
    dir_path = "data/documents_split"
    
    preprocessor = TextPreprocessor(method=method)
    inc_matrix = IncidenceMatrix()
    inv_index = InvertedIndex()
    pos_index = PositionalIndex()
    ranker = Ranker()
    spell_checker = SpellChecker()
    query_expander = QueryExpander()
    snippet_gen = SnippetGenerator(documents_dir=dir_path)
    
    documents = {}
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
        if not files:
            return None, "No .txt files found in data/documents_split."
            
        for filename in files:
            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    tokens = preprocessor.process(text)
                    documents[filename] = tokens
            except Exception as e:
                pass # skip unreadable files
    else:
        return None, f"Directory {dir_path} not found. Please run the data splitter first."
        
    if not documents:
        return None, "No documents parsed."
        
    # Build Indices
    inc_matrix.build_index(documents)
    inv_index.build_index(documents)
    pos_index.build_index(documents)
    ranker.compute_tf_idf(documents)
    
    # Load vocabulary into spell checker
    spell_checker.load_vocabulary(ranker.vocabulary)
    
    return {
        "preprocessor": preprocessor,
        "inc_matrix": inc_matrix,
        "inv_index": inv_index,
        "pos_index": pos_index,
        "ranker": ranker,
        "spell_checker": spell_checker,
        "query_expander": query_expander,
        "snippet_gen": snippet_gen,
        "num_docs": len(documents)
    }, "Success"

with st.spinner(f"Loading documents and building indices using {preprocessing_method}..."):
    engine, msg = load_and_build_engine(preprocessing_method)

if not engine:
    st.error(msg)
    st.stop()

st.sidebar.success(f"Indices built successfully for {engine['num_docs']} documents!")

# --- 4. Search UI ---
st.divider()
query = st.text_input("Enter your search query (supports wildcards like *):", placeholder="e.g. machine learning", value="")

if st.button("Search", type="primary") or query:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        # Check spelling first
        raw_tokens = query.split()
        corrections = engine["spell_checker"].get_corrections(raw_tokens)
        
        if corrections:
            corrected_query = " ".join([corrections.get(w, w) for w in raw_tokens])
            st.warning(f"Did you mean: **{corrected_query}** ?")
            
        # Preprocess Query
        query_tokens = engine["preprocessor"].process(query)
        st.write(f"**Processed Tokens:** `{query_tokens}`")
        
        # Query Expansion
        if enable_query_expansion:
            expanded_tokens = engine["query_expander"].expand_query(query_tokens)
            st.info(f"**Expanded Tokens (Synonyms):** `{expanded_tokens}`")
            search_tokens = expanded_tokens
        else:
            search_tokens = query_tokens
            
        # Define Tabs for clean organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Ranked Retrieval", 
            "Exact Phrase", 
            "Inverted Index", 
            "Incidence Matrix",
            "Positional Index"
        ])
        
        # --- TAB 1: Ranked Retrieval ---
        with tab1:
            st.subheader("Ranked Results (TF-IDF & Cosine Similarity)")
            ranked_docs = engine["ranker"].rank_documents(search_tokens)
            if ranked_docs:
                for rank, (doc, score) in enumerate(ranked_docs[:10], start=1):
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])
                        col1.metric(label=f"Rank #{rank}", value=f"{score:.4f}")
                        col2.markdown(f"**{doc}**")
                        snippet = engine["snippet_gen"].get_snippet(doc, search_tokens)
                        col2.caption(f"_{snippet}_")
            else:
                st.info("No documents match the query.")
                
        # --- TAB 2: Exact Phrase ---
        with tab2:
            st.subheader(f"Exact Phrase Match: '{query}'")
            phrase_matched_docs = engine["pos_index"].search_phrase(search_tokens)
            if phrase_matched_docs:
                for doc in phrase_matched_docs[:10]:
                    with st.container(border=True):
                        st.markdown(f"**{doc}**")
                        snippet = engine["snippet_gen"].get_snippet(doc, search_tokens)
                        st.caption(f"_{snippet}_")
            else:
                st.info("No exact phrase matches found in any document.")
                
        # --- TAB 3: Inverted Index ---
        with tab3:
            st.subheader("Inverted Index Search (Boolean AND)")
            inv_matched_docs = engine["inv_index"].search(search_tokens)
            if inv_matched_docs:
                st.write(f"Found {len(inv_matched_docs)} matching documents.")
                st.write(inv_matched_docs[:10])
            else:
                st.info("No matches found.")
                
            st.markdown("### Internal Structure (Posting Lists)")
            for term in search_tokens:
                posting_list = list(engine["inv_index"].index.get(term, set()))
                with st.expander(f"Term: '{term}' (Appears in {len(posting_list)} docs)"):
                    st.write(posting_list)
                    
        # --- TAB 4: Incidence Matrix ---
        with tab4:
            st.subheader("Incidence Matrix Search (Boolean AND)")
            inc_matched_docs = engine["inc_matrix"].search(search_tokens)
            if inc_matched_docs:
                st.write(f"Found {len(inc_matched_docs)} matching documents.")
            else:
                st.info("No matches found.")
                
            st.markdown("### Internal Structure (Binary Matrix)")
            df = engine["inc_matrix"].df
            if df is not None:
                valid_terms = [t for t in search_tokens if t in df.index]
                if valid_terms:
                    sub_df = df.loc[valid_terms]
                    all_match_docs = sub_df.columns[(sub_df == 1).all(axis=0)].tolist()
                    any_match_docs = [doc for doc in sub_df.columns[(sub_df == 1).any(axis=0)] if doc not in all_match_docs]
                    active_docs = all_match_docs + any_match_docs
                    
                    if active_docs:
                        st.dataframe(sub_df[active_docs[:50]], width="stretch")
                        st.caption("Showing documents containing at least one query term.")
                    else:
                        st.info("No documents contain these terms.")
                else:
                    st.info("Terms not in vocabulary.")
                    
        # --- TAB 5: Positional Index ---
        with tab5:
            st.subheader("Positional Index Mapping")
            for term in search_tokens:
                pos_list = engine["pos_index"].index.get(term, {})
                with st.expander(f"Term: '{term}' (Appears in {len(pos_list)} docs)"):
                    st.json(pos_list)
                    
