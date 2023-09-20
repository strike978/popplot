import streamlit as st
import pandas as pd
from scipy.cluster.hierarchy import linkage
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
import datetime

# Initialize session state attributes
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""

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
content_after_comma_set = set()
for file in selected_files:
    data = read_data_file(data_files[file])
    for line in data:
        content_after_comma = ",".join(line.split(',')[1:])
        if content_after_comma not in content_after_comma_set:
            selected_data.append(line)
            content_after_comma_set.add(content_after_comma)

# Get the populations already in the textbox
# populations_in_textbox = [line.split(
#     ',')[0] for line in st.session_state.textbox_content.strip().split('\n')]

# # Create a filtered list of available populations
# available_populations = [pop for pop in selected_data if pop.split(
#     ',')[0] not in populations_in_textbox]


# Get the populations already in the textbox
populations_in_textbox = [line.split(',')[1] if len(line.split(
    ',')) > 1 else '' for line in st.session_state.textbox_content.strip().split('\n')]

# Create a filtered list of available populations based on content after the comma
available_populations = [pop for pop in selected_data if pop.split(
    ',')[1] not in populations_in_textbox]


group_pop_toggle = st.toggle('Group Populations')


# Group populations with the same word before the first ":" when toggle is enabled
grouped_populations = {}
if group_pop_toggle:
    for pop in available_populations:
        parts = pop.split(',')
        if len(parts) > 1:
            key = parts[0].split(':')[0]  # Get the part before the first ":"
            if key not in grouped_populations:
                grouped_populations[key] = []
            grouped_populations[key].append(pop)

# Create a Selectbox to display populations based on the toggle
if group_pop_toggle:
    population_options = list(grouped_populations.keys())
else:
    population_options = available_populations

selected_option_index = st.selectbox(
    "Populations:",
    range(len(population_options)),
    format_func=lambda i: population_options[i].split(',')[0]
)

    # Check if grouping of populations is enabled
if group_pop_toggle:
    # Check if a valid index is selected and within the range of population_options
    if selected_option_index is not None and selected_option_index < len(population_options):
        # If valid, select the corresponding grouped populations
        selected_option = grouped_populations[population_options[selected_option_index]]
    else:
        # If not valid, set selected_option as an empty list
        selected_option = []
else:
    # If grouping is not enabled, check if a valid index is selected
    if selected_option_index is not None:
        # If valid, select the corresponding population option as a list
        selected_option = [population_options[selected_option_index]]
    else:
        # If not valid, set selected_option as an empty list
        selected_option = []

# Create a button to add the selected option to the Textbox
if st.button("Add Population"):
    if selected_option:
        for pop in selected_option:
            if pop not in st.session_state.textbox_content:
                st.session_state.textbox_content += "\n" + pop
        st.experimental_rerun()


# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in G25 coordinates format:',
                          st.session_state.textbox_content.strip(), height=300, key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    # Fixes issue with text reverting if changed twice?
    st.experimental_rerun()

# Generate a unique file name based on the current date and time
current_datetime = datetime.datetime.now()
file_name = f"data_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

col1, col2, col3 = st.columns(3)

with col1:
    plot_dendrogram = st.button('Plot Dendrogram')
with col2:
    plot_2d_pca = st.button('Plot PCA')
with col3:
    st.download_button(
        label="💾 Save Data",
        data=data_input,
        key="download_data",
        file_name=file_name,
    )


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

# st.caption(
#     "Run Admixture Analysis with [admixtr](http://admixtr.com)")
