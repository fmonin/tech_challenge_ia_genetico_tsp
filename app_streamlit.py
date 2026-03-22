from __future__ import annotations

import traceback

import streamlit as st

from llm_ui_service import RouteLLMService
from route_context_loader import list_example_questions


# Configuração geral da página.
st.set_page_config(
    page_title="Consulta Inteligente de Rotas com LLM",
    page_icon="🚚",
    layout="centered",
)


@st.cache_resource
# Esse cache mantém a instância do serviço viva entre interações da sessão.
def get_service() -> RouteLLMService:
    return RouteLLMService()


# Como estamos em Streamlit, centralizar o estado ajuda a manter a conversa.
def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "routes_ready" not in st.session_state:
        st.session_state.routes_ready = False
    if "warmup_error" not in st.session_state:
        st.session_state.warmup_error = None


def warmup_routes(service: RouteLLMService) -> None:
    try:
        with st.spinner("Preparando contexto das rotas..."):
            service.load_route_results()
        st.session_state.routes_ready = True
        st.session_state.warmup_error = None
    except Exception as exc:  # noqa: BLE001
        st.session_state.routes_ready = False
        st.session_state.warmup_error = str(exc)


def render_summary(service: RouteLLMService) -> None:
    st.subheader("Resumo das rotas")

    if st.button("Carregar resumo", use_container_width=True):
        try:
            with st.spinner("Carregando resumo..."):
                st.session_state.summary = service.get_summary()
        except Exception:
            st.session_state.summary = None
            st.warning("Não foi possível carregar o resumo das rotas agora.")

    if st.session_state.summary:
        summary = st.session_state.summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Veículos", summary["vehicles_used"])
        col2.metric("Paradas", summary["total_stops"])
        col3.metric("Distância (km)", f"{summary['total_distance_km']:.1f}")


# Trata a pergunta digitada pelo usuário.
def handle_user_prompt(service: RouteLLMService, prompt: str) -> None:
    try:
        with st.spinner("Consultando a LLM..."):
            answer = service.ask_question(prompt)
    except Exception as exc:  # noqa: BLE001
        answer = f"Não foi possível consultar a LLM neste momento.\n\nDetalhes: {exc}"

    st.session_state.history.insert(0, {"question": prompt, "answer": answer})


# Bloco principal da tela.
def main() -> None:
    init_session_state()
    service = get_service()

    st.title("Consulta de Rotas com IA")
    st.caption("Selecione uma pergunta sugerida ou escreva uma pergunta personalizada.")
    st.caption(
        "Na primeira execução, o sistema precisa montar as rotas. "
        "Use o botão abaixo para preparar o contexto antes de perguntar."
    )

    if st.button("Preparar contexto de rotas", use_container_width=True):
        warmup_routes(service)

    if st.session_state.routes_ready:
        st.success("Contexto carregado. As próximas perguntas respondem mais rápido.")
    elif st.session_state.warmup_error:
        st.error(f"Falha ao preparar contexto: {st.session_state.warmup_error}")
    else:
        st.info("Contexto ainda não carregado nesta sessão.")

    try:
        suggestions = list_example_questions()
    except Exception:
        suggestions = []

    if not suggestions:
        suggestions = ["Qual é a melhor rota do veículo 1?"]

    with st.form("question_form", clear_on_submit=False):
        selected_question = st.selectbox(
            "Perguntas sugeridas",
            options=suggestions,
            index=0,
        )
        custom_question = st.text_input(
            "Ou digite sua pergunta",
            placeholder="Ex.: Qual veículo teve a maior distância total?",
        )
        submitted = st.form_submit_button("Perguntar", use_container_width=True)

    if submitted:
        prompt = custom_question.strip() or selected_question
        if not st.session_state.routes_ready:
            warmup_routes(service)
        if not st.session_state.routes_ready:
            st.warning("Não foi possível preparar as rotas. Corrija o erro e tente novamente.")
            return
        handle_user_prompt(service, prompt)
        st.rerun()

    st.divider()
    render_summary(service)

    if st.session_state.history:
        latest = st.session_state.history[0]
        st.subheader("Resposta")
        st.markdown(f"**Pergunta:** {latest['question']}")
        st.markdown(latest["answer"])

        with st.expander("Histórico"):
            for item in st.session_state.history:
                st.markdown(f"**Pergunta:** {item['question']}")
                st.markdown(item["answer"])
                st.divider()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        st.error("Ocorreu um erro inesperado ao iniciar a aplicação.")
        st.code("".join(traceback.format_exception_only(type(exc), exc)).strip())
