import streamlit as st
import pandas as pd
from scipy.cluster.hierarchy import linkage
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA

# Initialize session state attributes
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""
if 'deleted_content' not in st.session_state:
    st.session_state.deleted_content = ""

# Setting the layout of the page to wide and the title of the page to PopPlot
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🧬")
st.header('Pop:green[Plot]')

# tab1, tab2 = st.tabs(["Data", "Plot"])

# with tab1:
# Read data from Modern Ancestry.txt
with open("Modern Ancestry.txt") as f:
    ancestry_list = [line.strip() for line in f]

# Create a Selectbox to display content before the first comma
selected_option_index = st.selectbox(
    "Select a population",
    range(len(ancestry_list)),
    format_func=lambda i: ancestry_list[i].split(',')[0]
)

# Create a button to add the entire selected option to the Textbox
if st.button("Add Population"):
    if selected_option_index is not None:
        selected_option = ancestry_list[selected_option_index]
        if selected_option not in st.session_state.textbox_content:
            st.session_state.textbox_content += "\n" + selected_option

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in G25 coordinates format:',
                          st.session_state.textbox_content.strip(), height=300, key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.deleted_content = ""
    st.session_state.textbox_content = data_input.strip()


# This code is creating two columns in the Streamlit app interface. The first column (`col1`) has a
# width of 1 and the second column (`col2`) has a width of 10.
col1, col2 = st.columns([1, 10])

with col1:
    plot_dendrogram = st.button('Plot Dendrogram')
with col2:
    plot_pca = st.button('Plot PCA')


if plot_dendrogram:
    with st.spinner("Creating Dendrogram..."):
        if data_input:
            # Remove leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())

            data = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None).iloc[:, 1:]
            populations = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0])[0]

            if not data.empty and len(populations) >= 2:
                labels = [i for i in populations]
                height = max(20 * len(populations), 500)
                fig = ff.create_dendrogram(
                    data,
                    orientation="right",
                    labels=labels,
                    linkagefun=lambda x: linkage(x, method="ward"),
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

if plot_pca:
    with st.spinner("Creating PCA Plot..."):
        if data_input:
            # Remove leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())

            # Read the data and select only the appropriate columns (PCA1 and PCA2)
            data = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0, 1, 2]).rename(columns={1: 'PCA1', 2: 'PCA2'})
            populations = data[0]

            if not data.empty and len(populations) >= 2:
                # Create a 2D scatter plot with labels
                fig = px.scatter(data, x='PCA1', y='PCA2', color=populations, title='',
                                 text=populations, labels={'color': 'Populations'})  # Use populations as labels

                # Customize hover text to show only the label (population name)
                fig.update_traces(textposition='top center',
                                  hovertemplate='%{text}')

                # Change the legend title to "Populations"
                fig.update_layout(legend_title_text='Populations')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Please add at least 2 populations before plotting.")
