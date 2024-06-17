# Imports
import os
import re
import asyncio
import streamlit as st
from model import get_rag_chain, get_cached_rag_chain, reset_rag_chain

# Start event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# RAG chain variable
rag_chain = get_cached_rag_chain()

# Init state
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
if 'reset' not in st.session_state:
    st.session_state['reset'] = False

# Title
st.set_page_config(page_title='Article Unraveler', page_icon=':sparkles:')
st.title("Article Unraveler")
st.subheader('Helps you research online articles through Q&A.')

# Sidebar for inputs
st.sidebar.title("Article URLs")
option = st.sidebar.selectbox('Number of URLs:', range(1,9), index=1)
st.sidebar.markdown("# ")
st.sidebar.write("**Enter URLs:**")
urls = []
for i in range(option):
    if(st.session_state['reset']):
        urls.append(st.sidebar.text_input(f'URL {i+1}', value=' ', key=f'url-box-{i}'))
    else:
        urls.append(st.sidebar.text_input(f'URL {i+1}', key=f'url-box-{i}'))
if(st.session_state['reset']):
    st.session_state['reset'] = False

# Message placeholder
message_placeholder = st.empty()    

# Process URL inputs
if(st.session_state['processed']):
    # Processed message
    message_placeholder.write(':heavy_check_mark: Processing done. Ready to answer your questions!')
    # Add reset button
    reset_button = st.sidebar.button("Reset App")
    # Reset app if reset button clicked
    if(reset_button):
        st.session_state['processed'] = False
        st.session_state['reset'] = True
        reset_rag_chain()
        st.experimental_rerun()
else:
    # Add process button
    process_urls_button_clicked = st.sidebar.button("Process URLs")
    # Generate RAG chain if process button clicked
    if(process_urls_button_clicked):
        # Check values
        valid_urls = []
        for url in urls:
            if(url and len(url.strip())): valid_urls.append(url.strip())
        if(len(valid_urls) == 0):
            # Set error message
            st.sidebar.warning('Invalid URLs')
        else:
            # Set placeholder value
            message_placeholder.write('Processing articles... :gear: ')
            # Get RAG chain
            rag_chain = get_rag_chain(valid_urls)
            # Update placeholder
            message_placeholder.write(':heavy_check_mark: Processing done. Ready to answer your questions!')
            # Update state and re render
            st.session_state['processed'] = True
            st.experimental_rerun()

# Query box
if(st.session_state['processed']):
    st.markdown('####')
    query = st.text_input("Question: ")
    if(query):
        answer = rag_chain.invoke(query)
        st.subheader("Answer")
        st.write(answer)
else:
    message_placeholder.write('Enter URLs and click "Process URLs" to get started.')