import os
import time
import tempfile
import streamlit as st
from credentials import (
    LLAMA_CLOUD_API_KEY, GROQ_API_KEY, HF_TOKEN,
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD,
)

os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Cache",
    page_icon="⚡",
    layout="wide",
)

# ── Cached resources (initialised once per session) ──────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_vectorizer():
    from redisvl.utils.vectorize import HFTextVectorizer
    return HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Connecting to Redis…")
def get_cache(distance_threshold: float = 0.2):
    from redisvl.extensions.cache.llm import SemanticCache
    redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
    return SemanticCache(
        vectorizer=get_vectorizer(),
        distance_threshold=distance_threshold,
        redis_url=redis_url,
    )

@st.cache_resource(show_spinner="Loading LLM…")
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [("hits", 0), ("misses", 0), ("history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Semantic Cache")
    st.caption("Redis · LangChain · HuggingFace")
    st.caption("Author: Clement T. Okolo")
    st.divider()

    # Connection status
    try:
        import redis as _redis
        r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        r.ping()
        st.success("Redis connected", icon="🟢")
    except Exception as e:
        st.error(f"Redis error: {e}", icon="🔴")

    st.divider()

    # Distance threshold — clear cached SemanticCache if changed
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.05, max_value=0.50, value=0.20, step=0.05,
        help="Maximum vector distance for a cache hit. Lower = stricter match.",
    )
    if st.session_state.get("threshold") != threshold:
        get_cache.clear()
        st.session_state["threshold"] = threshold

    st.divider()

    # Cache stats
    total = st.session_state.hits + st.session_state.misses
    hit_rate = (st.session_state.hits / total * 100) if total else 0
    col1, col2 = st.columns(2)
    col1.metric("Cache hits", st.session_state.hits)
    col2.metric("Cache misses", st.session_state.misses)
    st.progress(hit_rate / 100, text=f"Hit rate: {hit_rate:.0f}%")

    if st.button("Reset stats"):
        st.session_state.hits = 0
        st.session_state.misses = 0
        st.rerun()

    st.divider()

    # ── Populate cache from PDF ───────────────────────────────────────────────
    with st.expander("📄 Populate cache from PDF", expanded=False):
        st.caption(
            "Upload a PDF to parse it with LlamaCloud, extract FAQs with the "
            "Groq LLM, and load them into the cache. This may take a few minutes."
        )
        pdf_source = st.radio(
            "PDF source",
            ["Use existing (data/ folder)", "Upload a new PDF"],
            index=0,
        )
        uploaded_file = None
        if pdf_source == "Upload a new PDF":
            uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

        auto_store = st.checkbox("Store new LLM responses in cache", value=True)

        if st.button("🚀 Populate cache", use_container_width=True):
            _populate_clicked = True
        else:
            _populate_clicked = False

    if _populate_clicked:
        with st.spinner("Parsing PDF and generating FAQs…"):
            try:
                from llama_cloud import LlamaCloud
                from llama_index.core import Document
                from llama_index.core.node_parser import MarkdownNodeParser
                from langchain_core.output_parsers import JsonOutputParser
                from langchain_core.prompts import PromptTemplate
                from pydantic import BaseModel, Field
                from typing import List

                llama_client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

                # Resolve PDF path
                if pdf_source == "Upload a new PDF" and uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name
                else:
                    pdf_path = "./data/2022-chevrolet-colorado-ebrochure.pdf"

                # Upload & parse
                file_obj = llama_client.files.create(file=pdf_path, purpose="parse")
                result = llama_client.parsing.parse(
                    file_id=file_obj.id,
                    tier="agentic",
                    version="latest",
                    expand=["markdown"],
                )
                documents = [
                    Document(
                        text=page.markdown,
                        metadata={"page_label": str(i + 1)},
                    )
                    for i, page in enumerate(result.markdown.pages)
                ]

                # Chunk into nodes
                nodes = MarkdownNodeParser().get_nodes_from_documents(documents)

                # Define FAQ schema
                class PromptResponse(BaseModel):
                    prompt: str = Field(description="Question about the document.")
                    response: str = Field(description="Grounded answer from the document.")

                class FAQs(BaseModel):
                    pairs: List[PromptResponse] = Field(description="List of FAQ pairs.")

                json_parser = JsonOutputParser(pydantic_object=FAQs)
                llm = get_llm()
                prompt_template = PromptTemplate(
                    template=(
                        "Extract as many FAQ prompt/response pairs as possible from the "
                        "document context below. Focus on factual data.\n\n"
                        "{format_instructions}\n\nDocument Context:\n{doc}\n"
                    ),
                    input_variables=["doc"],
                    partial_variables={"format_instructions": json_parser.get_format_instructions()},
                )
                chain = prompt_template | llm | json_parser

                all_faqs = []
                progress = st.progress(0, text="Extracting FAQs from nodes…")
                for i, node in enumerate(nodes):
                    res = chain.invoke({"doc": node.text})
                    if res and res.get("pairs"):
                        all_faqs.extend(res["pairs"])
                    progress.progress((i + 1) / len(nodes), text=f"Node {i+1}/{len(nodes)}")

                # Embed & store
                vectorizer = get_vectorizer()
                prompts = [p["prompt"] for p in all_faqs]
                embeddings = vectorizer.embed_many(prompts)
                cache = get_cache(threshold)
                for i, entry in enumerate(all_faqs):
                    if "prompt" in entry and "response" in entry:
                        cache.store(
                            prompt=entry["prompt"],
                            response=entry["response"],
                            vector=embeddings[i],
                        )

                st.success(f"✅ Loaded {len(all_faqs)} FAQs into the cache.")
            except Exception as e:
                st.error(f"Failed to populate cache: {e}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask a question")
st.caption(
    "Semantically similar questions are answered from the Redis cache instantly. "
    "Cache misses fall back to the Groq LLM."
)

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Your question",
        placeholder="e.g. What engine options does the Chevy Colorado offer?",
        label_visibility="collapsed",
    )
    store_on_miss = st.checkbox("Store LLM response in cache on miss", value=True)
    submitted = st.form_submit_button("Ask ⚡", use_container_width=True)

if submitted and question.strip():
    cache = get_cache(threshold)
    llm = get_llm()

    start = time.perf_counter()
    cached_results = cache.check(question)
    elapsed = time.perf_counter() - start

    if cached_results:
        st.session_state.hits += 1
        answer = cached_results[0].get("response", "")
        score = cached_results[0].get("vector_distance", None)

        st.success("⚡ Cache hit", icon="✅")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Answer:**\n\n{answer}")
        with col2:
            st.metric("Response time", f"{elapsed * 1000:.1f} ms")
            if score is not None:
                st.metric("Vector distance", f"{score:.3f}")

    else:
        st.session_state.misses += 1
        with st.spinner("Cache miss — querying LLM…"):
            llm_start = time.perf_counter()
            llm_response = llm.invoke(question).content
            llm_elapsed = time.perf_counter() - llm_start

        st.warning("🔄 Cache miss — answered by LLM", icon="🤖")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Answer:**\n\n{llm_response}")
        with col2:
            st.metric("Response time", f"{llm_elapsed * 1000:.0f} ms")

        if store_on_miss:
            vectorizer = get_vectorizer()
            embedding = vectorizer.embed(question)
            cache.store(prompt=question, response=llm_response, vector=embedding)
            st.caption("💾 Stored in cache for future queries.")

    # Append to history
    st.session_state.history.append({
        "question": question,
        "hit": bool(cached_results),
    })

# ── Query history ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander("📋 Query history", expanded=False):
        for item in reversed(st.session_state.history):
            badge = "✅ hit" if item["hit"] else "🤖 miss"
            st.markdown(f"- {badge} — {item['question']}")
