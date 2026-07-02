import streamlit as st

from src.utils.db_utils import ensure_schema
from src.utils.etl_loader import get_sqlite_stats, load_excel_into_sqlite
from src.rag.utils import build_rag_corpus_complete


def show_data_loader() -> None:
    """Streamlit page for the Varun-only SQLite and RAG pipeline."""
    st.title("Data Ingestion Dashboard")
    st.markdown(
        """
        ## Load and index supply-chain data
        Initialize SQLite and ChromaDB from Varun's electronics workbook.
        Beauty/FMCG data is explicitly excluded.
        """
    )

    ensure_schema()
    left, right = st.columns(2)

    with left:
        st.subheader("SQLite Database")
        if st.button("Load Excel to SQLite"):
            with st.spinner("Loading all workbook sheets into SQLite..."):
                try:
                    record_count = load_excel_into_sqlite(flush_existing=True)
                    st.success(f"Loaded {record_count:,} order records into SQLite")
                except Exception as exc:
                    st.error(f"SQLite build failed: {exc}")

        if st.button("Show SQLite Statistics"):
            try:
                st.json(get_sqlite_stats())
            except Exception as exc:
                st.warning(f"Could not read database statistics: {exc}")

    with right:
        st.subheader("ChromaDB Vector Store")
        if st.button("Build Electronics RAG Corpus"):
            with st.spinner("Building the electronics semantic index..."):
                try:
                    results = build_rag_corpus_complete(flush_existing=True)
                    st.success("RAG corpus built")
                    st.json(results)
                except Exception as exc:
                    hint = ""
                    err = str(exc)
                    if "SSL" in err or "CERTIFICATE" in err:
                        hint = (
                            " Hugging Face download blocked by TLS — set "
                            "HF_INSECURE_SSL=1 and restart Streamlit, or "
                            "run the build once from a terminal with network access."
                        )
                    elif "sentence_transformers" in err:
                        hint = (
                            " Upgrade sentence-transformers (>=5.5.1) and set "
                            "EMBEDDING_MODEL_PATH to the Hugging Face repo "
                            "(default: mathurvarun84/supply-chain-embeddings)."
                        )
                    st.error(f"ChromaDB build failed: {exc}.{hint}")

    st.markdown("---")
    st.subheader("Data Summary")
    try:
        stats = get_sqlite_stats()
        if not stats.get("database_exists"):
            st.info("No database loaded yet.")
            return

        first, second, third = st.columns(3)
        with first:
            st.metric(
                "Order Records",
                stats.get("tables", {}).get("lite_master", 0),
            )
        with second:
            st.metric(
                "Semiconductor Signals",
                stats.get("tables", {}).get("semiconductor_signals", 0),
            )
        with third:
            st.metric("Unique Products", stats.get("unique_products", 0))

        st.info(f"Date range: {stats.get('date_range', 'No data')}")
        st.caption("Categories: " + ", ".join(stats.get("categories", [])))
    except Exception:
        st.info("Load the SQLite database to see its summary.")


if __name__ == "__main__":
    show_data_loader()
