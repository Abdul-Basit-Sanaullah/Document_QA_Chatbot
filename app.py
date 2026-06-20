import streamlit as st
import transformers

st.write("Transformers Version:")
st.write(transformers.__version__)

from transformers.pipelines import SUPPORTED_TASKS

st.write("Supported Tasks:")
st.write(list(SUPPORTED_TASKS.keys()))
