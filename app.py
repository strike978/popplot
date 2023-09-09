import pandas as pd
from scipy.cluster.hierarchy import linkage
import streamlit as st
import io
import plotly.figure_factory as ff

# Setting the layout of the page to wide and the title of the page to G25 Dendrograms
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🧬")
st.header('Pop:green[Plot]')

tab1, tab2 = st.tabs(["Data", "Plot"])

if 'deleted_content' not in st.session_state:
    st.session_state.deleted_content = ""

# Initialize selectbox_options and ancestry_list outside the block
selectbox_options = []
ancestry_list = []

# Initialize last_data_input attribute
if 'last_data_input' not in st.session_state:
    st.session_state.last_data_input = None

with tab1:
    if 'textbox_content' not in st.session_state:
        st.session_state.textbox_content = ""

    # Read data from Modern Ancestry.txt
    with open("Modern Ancestry.txt") as f:
        ancestry_list = [line.strip() for line in f]

        # Extract the content before the first comma for the Selectbox options
        selectbox_options = [line.split(
            ',')[0] if ',' in line else line for line in ancestry_list]

    # Create a Selectbox to display content before the first comma
    selected_option_index = st.selectbox(
        "Select a population",
        range(len(selectbox_options)),
        format_func=lambda i: selectbox_options[i]
    )

    # Create a button to add the entire selected option to the Textbox
    if st.button("Add Population"):
        if selected_option_index is not None:  # Check if a valid option is selected
            selected_option = ancestry_list[selected_option_index]
            if 'textbox_content' in st.session_state:
                # Check if the selected option is not already in the Textbox, not in the deleted content,
                # and not an empty line
                if selected_option not in st.session_state.textbox_content and selected_option not in st.session_state.deleted_content and selected_option.strip():
                    if st.session_state.textbox_content:
                        st.session_state.textbox_content += "\n" + selected_option
                    else:
                        st.session_state.textbox_content = selected_option

    # Display the Textbox with the entire selected options
    data_input = st.text_area('Enter data in G25 coordinates format:',
                              st.session_state.textbox_content, height=400, key='textbox_input')

    # Check if the Textbox content has changed manually and clear session state if it has
    if data_input != st.session_state.textbox_content:
        st.session_state.deleted_content = ""
        st.session_state.textbox_content = data_input

with tab2:
    plot_button = st.button("Plot")  # Add a Plot button

    if plot_button:
        # Only plot when the "Plot" button is clicked
        if data_input:
            # Remove any leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())
            st.session_state.textbox_content = cleaned_data_input  # Update the session state

            data = pd.read_csv(io.StringIO(cleaned_data_input),
                               header=None).iloc[:, 1:]
            populations = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0])[0]

            # Check if data_input is empty or contains less than 2 populations
            if not data.empty and len(populations) >= 2:
                labels = [i for i in populations]
                height = max(20 * len(populations), 500)
                fig = ff.create_dendrogram(
                    data,
                    orientation="right",
                    labels=labels,
                    linkagefun=lambda x: linkage(x, method="ward"),
                    # color_threshold=0.07,
                )
                fig.update_layout(
                    height=height,
                    yaxis={'side': 'right'}
                )
                fig.update_yaxes(
                    automargin=True,
                    range=[0, len(populations)*10]
                )

                st.plotly_chart(fig, theme=None, use_container_width=True)
            else:
                st.warning(
                    "Please add at least 2 populations before plotting.")
