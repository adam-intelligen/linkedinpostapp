"""Streamlit app to generate LinkedIn posts."""

# Import from standard library
import logging
import random
import re
import pandas as pd

# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components
from snowflake.snowpark.session import Session

# Import modules
import oai

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)


# Define functions
def generate_text(topic: str, style: str = ""):
    """Generate LinkedIn Post"""
    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before generating another post."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return

    st.session_state.post = ""
    st.session_state.image = ""
    st.session_state.text_error = ""

    if not topic:
        st.session_state.text_error = "Please enter a topic"
        return

    with text_spinner_placeholder:
        with st.spinner("Please wait while your LinkedIn post is being generated..."):
            style_prompt = f"{style} " if style else ""
            prompt = f"Write a {style_prompt} LinkedIn Post about the Snowflake data platform feature {topic}:\n\n include the hashtags datasuperhero and masteringsnowflake"

            openai = oai.Openai()
            flagged = openai.moderate(prompt)
            style_output = f", stype: {style}" if style else ""
            if flagged:
                st.session_state.text_error = "Input flagged as inappropriate."
                logging.info(f"Topic: {topic}{style_output}\n")
                return

            else:
                st.session_state.text_error = ""
                st.session_state.n_requests += 1
                st.session_state.post = (
                    openai.complete(prompt).strip().replace('"', "")
                )
                logging.info(
                    f"Topic: {topic}{style_output}\n"
                    f"Post: {st.session_state.post}"
                )


def generate_image(prompt: str):
    """Generate Post image."""
    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before generating another text or image."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return

    with image_spinner_placeholder:
        with st.spinner("Please wait while your image is being generated..."):
            openai = oai.Openai()
            prompt_wo_hashtags = re.sub("#[A-Za-z0-9_]+", "", prompt)
            processing_prompt = (
                "In less than 300 words create a detailed but brief description of an image that captures "
                f"the essence of the following text:\n{prompt_wo_hashtags}\n\n"
            )
            processed_prompt = (
                openai.complete(
                    prompt=processing_prompt, temperature=0.5, max_tokens=200
                )
                .strip()
                .replace('"', "")
                .split(".")[0]
                + "."
            )
            st.session_state.n_requests += 1
            st.session_state.image = openai.image(processed_prompt)
            logging.info(f"Post: {prompt}\nImage prompt: {processed_prompt}")
   

# Configure Streamlit page and state
st.set_page_config(page_title="LinkedIn Post", page_icon="❄️")

if "post" not in st.session_state:
    st.session_state.post = ""
if "image" not in st.session_state:
    st.session_state.image = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""
if "image_error" not in st.session_state:
    st.session_state.image_error = ""
if "feeling_lucky" not in st.session_state:
    st.session_state.feeling_lucky = False
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0

# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def create_session():
     return Session.builder.configs(st.secrets.snowflake).create()
session = create_session()
st.success("Connected to Snowflake!")

# Uses st.cache_data to only rerun when the query changes 
@st.cache_data
def load_sf_data(table_name):
    # Use the session object to query Snowflake
    session = create_session()  # Get the session object from cache
    table = session.table(table_name)
    table = table.collect()
    return table


# Load data from Snowflake and get a specific column as a DataFrame
df = load_sf_data('snowflake_features')  
sffeatures = [row[0] for row in df]

df = load_sf_data('style')  
style = [row[0] for row in df]

# Render Streamlit page
st.title("Generate LinkedIn Posts")
st.markdown(
    "This mini-app generates Snowflake related LinkedIn Posts using OpenAI's GPT-3 based [Davinci model](https://beta.openai.com/docs/models/overview) for texts and [DALL·E](https://beta.openai.com/docs/guides/images) for images."
)

#initialise the edited_style variable to the data from Snowflake
edited_style = style

#leverage the st.experimental_data_editor function to allow users to add their own styles
if st.checkbox('Add your own styles here:'):
    st.subheader('Available styles')
    edited_style = st.experimental_data_editor(style,  num_rows="dynamic")


topic = st.selectbox('Select a Snowflake feature...',sffeatures)

style = st.selectbox('Select a style...',edited_style)

col1, col2 = st.columns(2)
with col1:
    st.session_state.feeling_lucky = not st.button(
        label="Generate text",
        type="primary",
        on_click=generate_text,
        args=(topic, style),
    )


text_spinner_placeholder = st.empty()
if st.session_state.text_error:
    st.error(st.session_state.text_error)

if st.session_state.post:
    st.markdown("""---""")
    st.text_area(label="Post", value=st.session_state.post, height=200)
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.feeling_lucky:
            st.button(
                label="Regenerate text",
                type="secondary",
                on_click=generate_text,
                args=(random.choice(topic), random.choice(style)),
            )
        else:
            st.button(
                label="Regenerate text",
                type="secondary",
                on_click=generate_text,
                args=(topic, style),
            )

    if not st.session_state.image:
        st.button(
            label="Generate image",
            type="primary",
            on_click=generate_image,
            args=[st.session_state.post],
        )
    else:
        st.image(st.session_state.image)
        st.button(
            label="Regenerate image",
            type="secondary",
            on_click=generate_image,
            args=[st.session_state.post],
        )

    image_spinner_placeholder = st.empty()
    if st.session_state.image_error:
        st.error(st.session_state.image_error)

    st.markdown("""---""")
   

