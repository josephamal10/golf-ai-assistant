from __future__ import annotations

import os
import time

import httpx
import streamlit as st


API_URL = os.environ.get('GOLF_API_URL', 'http://127.0.0.1:8001')
ASK_TIMEOUT_SECONDS = 150.0                                # grounded answers can take ~50s
HEALTH_TTL_SECONDS = 30                                    # keeps reruns off the health endpoint

# One per category, all questions the retrieval eval covers.
EXAMPLE_QUESTIONS = [
    'What is the difference between stroke play and match play?',
    "What are the four men's major championships?",
    'What is a links golf course?',
    'What is the dimple pattern on a golf ball for?',
    'Who jointly writes the Rules of Golf?',
    'What is a mulligan?',
]

st.set_page_config(
    page_title='Golf AI Assistant',
    page_icon='⛳',
    layout='centered',
    initial_sidebar_state='expanded',
)


@st.cache_data(ttl=HEALTH_TTL_SECONDS, show_spinner=False)
def fetch_health() -> dict | None:
    """None means the API is unreachable, which disables the composer."""
    try:
        response = httpx.get(f'{API_URL}/health', timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def friendly_error(exc: httpx.HTTPStatusError) -> str:
    """Reads the API's status codes so a quota problem is not mistaken for a bug."""
    try:
        detail = str(exc.response.json().get('detail') or '')
    except ValueError:
        detail = ''
    code = exc.response.status_code
    if code == 429:
        return detail or 'The provider rate limit or daily quota is exhausted. Try again later.'
    if code == 503:
        return detail or 'The model provider is temporarily unavailable. Try again in a minute.'
    if code == 422:
        return 'That question is too short or too long.'
    return detail or f'The API returned HTTP {code}.'


def render_sources(sources: list[dict]) -> None:
    """Numbers match the [n] markers in the answer so any claim can be traced back."""
    if not sources:
        return
    with st.expander(f'Sources ({len(sources)})'):
        for source in sources:
            with st.container(border=True):
                st.markdown(f"**[{source['n']}]** [{source['title']}](<{source['url']}>)")
                if source.get('category'):
                    st.badge(source['category'], color='green')


def render_assistant(message: dict) -> None:
    """Shared by the live turn and the replayed history so both look identical."""
    if message.get('error'):
        st.error(message['content'])
        return
    st.markdown(message['content'])
    footnote = [
        part
        for part in (
            f"{message.get('provider')} · {message.get('model')}" if message.get('model') else '',
            f"{message['elapsed']:.1f}s" if message.get('elapsed') else '',
        )
        if part
    ]
    if footnote:
        st.caption(' — '.join(footnote))
    render_sources(message.get('sources') or [])


if 'messages' not in st.session_state:
    st.session_state.messages = []

health = fetch_health()

st.title('⛳ Golf AI Assistant')
st.caption(
    'Ask about rules, history, courses, tournaments, players, or equipment. '
    'Every answer is drawn from retrieved passages and cites the articles it used.'
)

if not st.session_state.messages and 'pending_question' not in st.session_state:
    st.markdown('##### Start with an example')
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(example, key=f'example-{index}', use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_assistant(message)
        else:
            st.markdown(message['content'])

typed = st.chat_input(
    'Ask about golf rules, history, courses, players, or equipment…',
    disabled=health is None,
)
question = typed or st.session_state.pop('pending_question', None)

if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        message: dict = {'role': 'assistant'}
        try:
            with st.spinner('Searching the knowledge base and drafting a cited answer…'):
                started = time.perf_counter()
                response = httpx.post(
                    f'{API_URL}/ask',
                    json={'question': question},
                    timeout=ASK_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
                elapsed = time.perf_counter() - started
            message.update(
                content=data.get('answer') or 'No answer returned.',
                sources=data.get('sources') or [],
                provider=data.get('provider', ''),
                model=data.get('model', ''),
                elapsed=elapsed,
            )
        except httpx.ConnectError:
            message.update(content=f'Cannot reach the API at {API_URL}.', error=True)
        except httpx.TimeoutException:
            message.update(
                content=f'No response within {ASK_TIMEOUT_SECONDS:.0f}s. Try again in a moment.',
                error=True,
            )
        except httpx.HTTPStatusError as exc:
            message.update(content=friendly_error(exc), error=True)
        render_assistant(message)

    st.session_state.messages.append(message)

# Drawn last, so controls reflect the conversation after this turn was appended.
# Sidebar placement is by container, not script order, so it still renders on the left.
with st.sidebar:
    st.markdown('### ⛳ Golf AI Assistant')
    st.caption('Phase 1 — static retrieval, grounded answers, always cited.')
    st.divider()

    if health:
        st.success('API online')
        st.markdown(
            f"**Generator**  \n{health.get('llm_provider', 'unknown')} · "
            f"`{health.get('llm_model', 'unknown')}`"
        )
    else:
        st.error('API offline')
        st.caption(f'Nothing answering at {API_URL}. Start it with:')
        st.code('uvicorn app.main:app --port 8001', language='bash')
        if st.button('Retry connection', use_container_width=True):
            fetch_health.clear()
            st.rerun()

    st.divider()
    if st.button(
        'Clear conversation',
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown('**Scope**')
    st.caption(
        'Answers come only from a static knowledge base of curated Wikipedia articles. '
        'Live scores, current rankings, and predictions are out of scope in Phase 1.'
    )
    st.caption('Source text from Wikipedia, CC BY-SA 4.0.')
