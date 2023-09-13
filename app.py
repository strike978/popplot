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

st.caption(
    'This site is optimized for desktop computers. You may experience some difficulty viewing it on a mobile device.')

# Define the available data files
data_files = {
    "Modern Era": "Modern Ancestry.txt",
    "Mesolithic and Neolithic": "Mesolithic and Neolithic.txt",
    "Bronze Age": "Bronze Age.txt",
    "Iron Age": "Iron Age.txt",
    "Migration Period": "Migration Period.txt",
    "Middle Ages": "Middle Ages.txt",
}

# Define a function to read data from a file with UTF-8 encoding


def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


# Create a multiselect checkbox to choose data files
selected_files = st.multiselect("Time Period:", list(
    data_files.keys()), default=["Modern Era"])

# Read data from selected files
selected_data = []
for file in selected_files:
    selected_data.extend(read_data_file(data_files[file]))

# Create a Selectbox to display content before the first comma
selected_option_index = st.selectbox(
    "Populations:",
    range(len(selected_data)),
    format_func=lambda i: selected_data[i].split(',')[0]
)

# Create a button to add the entire selected option to the Textbox
if st.button("Add Population"):
    if selected_option_index is not None:
        selected_option = selected_data[selected_option_index]
        if selected_option not in st.session_state.textbox_content:
            st.session_state.textbox_content += "\n" + selected_option

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in G25 coordinates format:',
                          st.session_state.textbox_content.strip(), height=300, key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.deleted_content = ""
    st.session_state.textbox_content = data_input.strip()
    # Fixes issue with text reverting if changed twice?
    st.experimental_rerun()


# The line `col1, col2, col3 = st.columns([1.2, 0.9, 11])` is creating three columns in the Streamlit
# app interface.
# col1, col2, col3 = st.columns([1.2, 0.9, 11])
col1, col2 = st.columns([1, 10])

with col1:
    plot_dendrogram = st.button('Plot Dendrogram')
with col2:
    plot_2d_pca = st.button('Plot PCA')
# with col3:
#     plot_3d_pca = st.button('Plot 3D PCA')


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

if plot_2d_pca:
    with st.spinner("Creating 2D PCA Plot..."):
        if data_input:
            # Remove leading/trailing whitespace and empty lines
            cleaned_data_input = "\n".join(
                line.strip() for line in data_input.splitlines() if line.strip())

            # Read the data and select all columns except the first one (which contains population labels)
            data = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None).iloc[:, 1:]

            populations = pd.read_csv(io.StringIO(
                cleaned_data_input), header=None, usecols=[0])[0]

            if not data.empty and len(populations) >= 2:
                # Perform PCA with all columns
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data)

                # Create a DataFrame for the PCA results
                pca_df = pd.DataFrame(
                    data=pca_result, columns=['PCA1', 'PCA2'])

                # Add the population labels back to the PCA DataFrame
                pca_df['Populations'] = populations

                # Create a 2D scatter plot with labels
                fig = px.scatter(pca_df, x='PCA1', y='PCA2', color='Populations',
                                 title='', text='Populations')

                # Customize hover text to show only the label (population name)
                fig.update_traces(textposition='top center',
                                  hovertemplate='%{text}')

                # Change the legend title to "Populations"
                fig.update_layout(legend_title_text='Populations')
                # Remove the axis labels
                fig.update_xaxes(title_text='')
                fig.update_yaxes(title_text='')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Please add at least 2 populations before plotting.")

# if plot_3d_pca:
#     with st.spinner("Creating 3D PCA Plot..."):
#         if data_input:
#             # Remove leading/trailing whitespace and empty lines
#             cleaned_data_input = "\n".join(
#                 line.strip() for line in data_input.splitlines() if line.strip())

#             # Read the data, excluding the first column (which contains population labels)
#             data = pd.read_csv(io.StringIO(
#                 cleaned_data_input), header=None).iloc[:, 1:]

#             populations = pd.read_csv(io.StringIO(
#                 cleaned_data_input), header=None, usecols=[0])[0]

#             if not data.empty and len(populations) >= 2:
#                 # Perform PCA with all columns
#                 pca = PCA(n_components=3)  # Perform 3D PCA
#                 pca_result = pca.fit_transform(data)

#                 # Create a DataFrame for the PCA results
#                 pca_df = pd.DataFrame(
#                     data=pca_result, columns=['PCA1', 'PCA2', 'PCA3'])

#                 # Add the population labels back to the PCA DataFrame
#                 pca_df['Populations'] = populations

#                 # Create a 3D scatter plot with labels
#                 fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color='Populations',
#                                     title='', text='Populations')

#                 # Customize hover text to show only the label (population name)
#                 fig.update_traces(textposition='top center',
#                                   hovertemplate='%{text}')

#                 # Change the legend title to "Populations"
#                 fig.update_layout(legend_title_text='Populations')
#                 # Remove the axis labels
#                 fig.update_layout(scene=dict(
#                     xaxis_title='', yaxis_title='', zaxis_title=''))

#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning(
#                     "Please add at least 2 populations before plotting.")