import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Afa shop! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Afa shop is a recommendation system that offers the best matches to the required products.
    **👈 Select a demo from the sidebar** to see some examples
    of what Afa shop can do!
    ### There are 2 rejimes implemented:
    1. Select the item from the validation set of the Shopee dataset as the input
    2. Type in your own text description of the desired item or upload the picture of it
"""
)
