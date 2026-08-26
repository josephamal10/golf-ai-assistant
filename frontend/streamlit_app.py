from __future__ import annotations

import os

import httpx
import streamlit as st


API_URL = os.environ.get('GOLF_API_URL', 'http://127.0.0.1:8000')

st.set_page_config(page_title='Golf AI Assistant', page_icon='⛳', layout='centered')
st.title('⛳ Golf AI Assistant')
st.caption('Phase 1 static RAG — answers grounded in retrieved golf knowledge, with citations.')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Sources'):
                for source in message['sources']:
                    st.markdown(
                        f"**[{source['n']}] {source['title']}** — "
                        f"{source['category']} · {source['source']} · {source['url']}"
                    )

question = st.chat_input('Ask about golf rules, history, courses, players, equipment...')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        try:
            response = httpx.post(
                f'{API_URL}/ask',
                json={'question': question},
                timeout=90.0,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get('answer') or 'No answer returned.'
            sources = data.get('sources') or []
            st.markdown(answer)
            if sources:
                with st.expander('Sources'):
                    for source in sources:
                        st.markdown(
                            f"**[{source['n']}] {source['title']}** — "
                            f"{source['category']} · {source['source']} · {source['url']}"
                        )
            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer,
                'sources': sources,
            })
        except httpx.ConnectError:
            st.error('API is not running. Start it with `uvicorn app.main:app --reload --port 8000`.')
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get('detail', exc.response.text)
            st.error(f'API error: {detail}')
