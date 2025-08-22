# app.py
import os
import streamlit as st

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

from rag_utils import (
    get_llm, get_embeddings,
    load_docs_from_files, split_documents,
    ensure_vectorstore, reset_vectorstore, retrieve_context
)
from prompts import SYSTEM_RAG_PROMPT, USER_RAG_INSTRUCTION, CITATION_FOOTER

load_dotenv()

st.set_page_config(page_title="Frontal — Agente de Neurologia", page_icon="🧠", layout="wide")

# Sidebar: Config
st.sidebar.title("⚙️ Configurações")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
model_choice = st.sidebar.selectbox("Modelo de Chat", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4-turbo"])
embed_choice = st.sidebar.selectbox("Modelo de Embedding", ["text-embedding-3-large", "text-embedding-3-small"])
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)

st.sidebar.markdown("---")
if st.sidebar.button("Limpar base vetorial"):
    reset_vectorstore()
    st.sidebar.success("Base vetorial removida. Recarregue documentos em 'Treinar base'.")

# Estado de sessão
if "history" not in st.session_state:
    st.session_state.history = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# Cabeçalho
st.title("Frontal — Agente de IA em Neurologia 🧠")
st.caption("Suporte a perguntas, raciocínio diagnóstico e exame neurológico com RAG. "
           "Este conteúdo é apenas educacional e não substitui o julgamento clínico.")

# TABS
tab_ask, tab_case, tab_train, tab_kb = st.tabs(["Perguntar", "Caso clínico", "Treinar base", "Base de conhecimento"])

# Preparar LLM/Embeddings sob demanda
def get_clients():
    llm = get_llm(api_key_input, model_choice, temperature)
    emb = get_embeddings(api_key_input, embed_choice)
    return llm, emb

# Função de resposta RAG genérica
def respond_with_rag(question: str, mode: str = "qa"):
    llm, emb = get_clients()
    # Carrega ou cria VS vazio
    from langchain_community.vectorstores import FAISS
    vs = None
    index_path = os.path.join("vectorstore", "faiss_index")
    if os.path.exists(index_path):
        try:
            vs = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
        except Exception:
            vs = None

    context, docs = retrieve_context(vs, question, k=6)
    system_prompt = SYSTEM_RAG_PROMPT
    # Histórico
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in st.session_state.history[-6:]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "Contexto recuperado (se houver):\n{context}"),
        ("user", USER_RAG_INSTRUCTION)
    ])

    chain = prompt | llm
    out = chain.invoke({
        "context": context if context else "Sem contexto relevante recuperado.",
        "question": question,
        "history": history_text if history_text else "Sem histórico relevante."
    })
    answer = out.content

    # Citações
    if docs:
        seen = set()
        sources_lines = []
        for d in docs:
            meta = d.metadata or {}
            source = meta.get("source", "Documento")
            page = meta.get("page", None)
            key = (source, page)
            if key not in seen:
                seen.add(key)
                if page is not None:
                    sources_lines.append(f"- {source} (p. {page})")
                else:
                    sources_lines.append(f"- {source}")
        if sources_lines:
            answer += "\n\n" + CITATION_FOOTER.format(sources="\n".join(sources_lines))

    # Atualiza histórico
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})

    return answer

with tab_train:
    st.subheader("Treinar base com artigos e livros")
    st.write("Carregue PDFs, DOCX ou TXT. O conteúdo será indexado localmente (FAISS).")
    uploads = st.file_uploader("Selecione arquivos", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    col_a, col_b = st.columns([1,1])
    with col_a:
        chunk_size = st.number_input("Tamanho do chunk", min_value=300, max_value=4000, value=1200, step=50)
    with col_b:
        chunk_overlap = st.number_input("Sobreposição", min_value=0, max_value=800, value=150, step=10)

    if st.button("Processar e Indexar"):
        if not api_key_input:
            st.error("Forneça sua OpenAI API Key no painel lateral.")
        elif not uploads:
            st.warning("Envie ao menos um arquivo.")
        else:
            with st.spinner("Carregando, segmentando e indexando..."):
                llm, emb = get_clients()
                docs = load_docs_from_files(uploads)
                if not docs:
                    st.error("Nenhum documento suportado foi detectado.")
                else:
                    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    vs = ensure_vectorstore(splits, emb, persist_dir="vectorstore")
                    st.session_state.vectorstore_ready = True
                    st.success(f"Indexados {len(splits)} chunks. Base pronta para consultas!")

with tab_ask:
    st.subheader("Pergunte ao Frontal")
    query = st.text_area("Sua pergunta (neurologia clínica, exames, condutas, fisiopatologia etc.)", height=120)
    if st.button("Responder", type="primary"):
        if not api_key_input:
            st.error("Forneça sua OpenAI API Key no painel lateral.")
        elif not query.strip():
            st.warning("Digite uma pergunta.")
        else:
            with st.spinner("Raciocinando..."):
                answer = respond_with_rag(query, mode="qa")
                st.markdown(answer)

with tab_case:
    st.subheader("Caso Clínico — Sugestão de Diagnósticos")
    st.write("Cole o caso (idade, início, tempo, comorbidades, sinais focais, exame neurológico, exames já feitos).")
    case_text = st.text_area("Caso clínico", height=220, placeholder="Ex.: Paciente 67a, HAS/DM, início súbito de hemiparesia D há 2 horas...")
    col1, col2 = st.columns([1,1])
    with col1:
        add_question = st.text_input("Pergunta específica (opcional)", placeholder="Ex.: principais diferenciais e próximos passos?")
    with col2:
        k_sources = st.number_input("Nº de trechos recuperados", min_value=1, max_value=12, value=6, step=1)

    if st.button("Sugerir diagnósticos"):
        if not api_key_input:
            st.error("Forneça sua OpenAI API Key no painel lateral.")
        elif not case_text.strip():
            st.warning("Descreva o caso clínico.")
        else:
            question = f"CASO CLÍNICO:\n{case_text}\n\nTAREFA: aplicar protocolo de diagnóstico diferencial, conforme instruções do sistema. " \
                       f"{'Pergunta específica: ' + add_question if add_question else ''}"
            with st.spinner("Analisando o caso e estruturando diferenciais..."):
                answer = respond_with_rag(question, mode="diagnostico")
                st.markdown(answer)

with tab_kb:
    st.subheader("Base de conhecimento e histórico")
    st.write("A base vetorial é local e persiste em ./vectorstore. Você pode limpar na barra lateral.")
    st.write("Histórico recente:")
    if st.session_state.history:
        for m in st.session_state.history[-8:]:
            role = "👤 Usuário" if m["role"] == "user" else "🤖 Frontal"
            with st.expander(role):
                st.markdown(m["content"])
    else:
        st.info("Sem histórico ainda.")

# Rodapé: Aviso
st.markdown("---")
st.caption("Aviso: Conteúdo educacional. Não substitui avaliação clínica, diretrizes oficiais ou protocolos locais.")
st.caption("Este agente foi desenvolvido pelo Dr Fernando Freua - distribuição gratuita.")