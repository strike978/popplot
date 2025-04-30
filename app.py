from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, cdist, squareform  # Add missing import for squareform
from sklearn.preprocessing import normalize
import streamlit as st
import pandas as pd
import io
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import io
from PIL import Image
import base64
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import MDS, Isomap


# Keep only these session state initializations
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""
if 'textbox_history' not in st.session_state:
    st.session_state.textbox_history = []
if 'redo_history' not in st.session_state:
    st.session_state.redo_history = []
if 'decomposition_method' not in st.session_state:
    st.session_state.decomposition_method = "PCA"
if 'linkage_method' not in st.session_state:
    st.session_state.linkage_method = "ward"
if 'distance_metric' not in st.session_state:
    st.session_state.distance_metric = "euclidean"

# Keep only these constants:
LINKAGE_METHOD = "ward"
DISTANCE_METRIC = "euclidean"

# Setting the layout of the page to wide and the title of the page to PopPlot
st.set_page_config(layout="wide", page_title="PopPlot", page_icon="🌎")
st.subheader('Pop:green[Plot]')

# Define the available data files
data_files = {
    "Modern Period": "Modern Ancestry.txt",
    "Stone Age": "Mesolithic and Neolithic.txt",
    "Bronze Age": "Bronze Age.txt",
    "Iron Age": "Iron Age.txt",
    "Migration Period": "Migration Period.txt",
    "Medieval Period": "Middle Ages.txt",
}

# Define a function to read data from a file with UTF-8 encoding


def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


# Create a multiselect checkbox to choose data files
selected_files = st.multiselect("Historical Period:", list(
    data_files.keys()), default=["Modern Period"])

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
populations_in_textbox = []
for line in st.session_state.textbox_content.strip().split('\n'):
    if line:  # Only process non-empty lines
        parts = line.split(',')
        if len(parts) > 1:
            # Store the data part (everything after the first comma)
            data_part = ','.join(parts[1:])
            populations_in_textbox.append(data_part)

# Create a filtered list of available populations based on content after the comma
available_populations = []
for pop in selected_data:
    parts = pop.split(',')
    if len(parts) > 1:
        data_part = ','.join(parts[1:])
        if data_part not in populations_in_textbox:
            available_populations.append(pop)

group_pop_toggle = st.checkbox('Group by Population')

# Group populations based on the word after the first ":" when the toggle is enabled
grouped_populations = {}
if group_pop_toggle:
    # First pass: collect all groups in order
    population_options = []
    seen_groups = set()

    # First, populate groups with available populations only
    for pop in available_populations:  # Changed from selected_data to available_populations
        parts = pop.split(',')
        if len(parts) > 1:
            name_parts = parts[0].split(':')
            if len(name_parts) > 1:
                # Take only first two parts if they exist
                for group in name_parts[:-1][:2]:
                    group = group.strip()
                    if group not in seen_groups:
                        population_options.append(group)
                        seen_groups.add(group)
                        grouped_populations[group] = []
            else:
                # Handle single-part names
                group = name_parts[0].strip()
                if group not in seen_groups:
                    population_options.append(group)
                    seen_groups.add(group)
                    grouped_populations[group] = []

    # Second pass: populate the groups
    for pop in available_populations:
        parts = pop.split(',')
        if len(parts) > 1:
            name_parts = parts[0].split(':')
            if len(name_parts) > 1:
                # Add to first two groups if they exist
                for group in name_parts[:-1][:2]:
                    group = group.strip()
                    if group in grouped_populations:
                        grouped_populations[group].append(pop)
            else:
                # Handle single-part names
                group = name_parts[0].strip()
                if group in grouped_populations:
                    grouped_populations[group].append(pop)

    # Remove empty groups and update population_options
    non_empty_groups = [
        group for group in population_options if grouped_populations[group]]
    population_options = non_empty_groups
    grouped_populations = {k: v for k, v in grouped_populations.items() if v}

else:
    population_options = available_populations

# Initialize session state for indices and callback
if 'last_group_index' not in st.session_state:
    st.session_state.last_group_index = 0
if 'last_individual_index' not in st.session_state:
    st.session_state.last_individual_index = 0
if 'selectbox_changed' not in st.session_state:
    st.session_state.selectbox_changed = False

# Define callback function for selectbox


def on_selectbox_change():
    st.session_state.selectbox_changed = True
    current_index = population_options.index(
        st.session_state.population_selectbox)
    if group_pop_toggle:
        st.session_state.last_group_index = current_index
    else:
        st.session_state.last_individual_index = current_index


# Use the appropriate stored index based on toggle state
default_index = st.session_state.last_group_index if group_pop_toggle else st.session_state.last_individual_index
default_index = min(default_index, len(
    population_options) - 1) if population_options else 0

# Modify the selectbox to use callback
selected_option = st.selectbox(
    "Populations:",
    population_options,
    format_func=lambda x: x.split(',')[0] if not group_pop_toggle else x,
    key='population_selectbox',
    index=default_index,
    on_change=on_selectbox_change
)

# Modify the selection logic
if group_pop_toggle:
    selected_populations = grouped_populations[selected_option]
else:
    selected_populations = [selected_option]

# Preserve the previous state of the textbox content
if 'previous_textbox_content' not in st.session_state:
    st.session_state.previous_textbox_content = ""

# Create a row of buttons for Add and Clear
# Make the first column smaller since Clear button is smaller
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("➕ Add"):
        if selected_populations:
            st.session_state.textbox_history.append(
                st.session_state.textbox_content)
            st.session_state.redo_history.clear()
            for pop in selected_populations:
                # Split by comma to separate population name from data
                parts = pop.split(',')
                if len(parts) > 1:
                    # Get the last part after colon for the population name
                    name_parts = parts[0].split(':')
                    display_name = name_parts[-1].strip()
                    # Get the data part
                    data_part = ','.join(parts[1:])
                    # Combine the display name with the data
                    formatted_pop = display_name + ',' + data_part

                    # Check if this data is already in the textbox
                    existing_data = [line.split(',', 1)[1] if ',' in line else ''
                                     for line in st.session_state.textbox_content.strip().split('\n')
                                     if line.strip()]

                    if data_part not in existing_data:
                        st.session_state.textbox_content += "\n" + formatted_pop.strip()

            # Get current index before calculating next
            current_index = population_options.index(selected_option)
            next_index = (current_index + 1) % len(population_options)
            if group_pop_toggle:
                st.session_state.last_group_index = next_index
            else:
                st.session_state.last_individual_index = next_index

            st.session_state.selectbox_changed = False
            st.rerun()

with col2:
    if st.button("🧹 Clear"):
        st.session_state.textbox_history.append(
            st.session_state.textbox_content)
        st.session_state.redo_history.clear()
        st.session_state.textbox_content = ""
        st.rerun()

# Commenting out Undo and Redo buttons
# with col3:
#     if st.button("↩️ Undo"):
#         if st.session_state.textbox_history:
#             st.session_state.redo_history.append(
#                 st.session_state.textbox_content)
#             st.session_state.textbox_content = st.session_state.textbox_history.pop()
#         st.rerun()

# with col4:
#     if st.button("↪️ Redo"):
#         if st.session_state.redo_history:
#             st.session_state.textbox_history.append(
#                 st.session_state.textbox_content)
#             st.session_state.textbox_content = st.session_state.redo_history.pop()
#         st.rerun()

# Add this function near the top of the file, after the imports


def parse_input_data(data_input):
    """Parse input data with mixed CSV/TSV formats."""
    parsed_lines = []
    for line in data_input.splitlines():
        if line.strip():
            # Count tabs and commas in the line
            tab_count = line.count('\t')
            comma_count = line.count(',')

            # Determine which delimiter splits the line into more fields
            if tab_count > comma_count:
                delimiter = '\t'
            else:
                delimiter = ','

            # Split the line and clean the fields - FIX THE BUG HERE
            fields = [field.strip() for field in line.split(delimiter)]
            parsed_lines.append(fields)

    # Convert to DataFrame
    if parsed_lines:
        df = pd.DataFrame(parsed_lines)
        # First column is population names, rest is data
        return df.iloc[:, 0], df.iloc[:, 1:].astype(float)
    return pd.Series(), pd.DataFrame()


# Add a function to calculate the recommended number of neighbors based on population count
def calculate_recommended_neighbors(population_count):
    """
    Calculate recommended number of neighbors for Isomap based on population count.
    Rules:
    - Minimum of 2 neighbors
    - For small datasets (3-10 populations): use half the population count
    - For medium datasets (11-30 populations): use one-third of the population count
    - For large datasets (>30 populations): use one-fourth of the population count
    - Maximum of 20 neighbors
    """
    if population_count <= 10:
        recommended = max(2, population_count // 2)
    elif population_count <= 30:
        recommended = max(5, population_count // 3)
    else:
        recommended = max(8, min(20, population_count // 4))
    
    return recommended

# Add a function to calculate the recommended perplexity based on population count
def calculate_recommended_perplexity(population_count):
    """
    Calculate recommended perplexity for t-SNE based on population count.
    Rules:
    - For small datasets (3-10 populations): use population count / 3
    - For medium datasets (11-30 populations): use population count / 4
    - For large datasets (>30 populations): use population count / 5
    - Minimum of 3, maximum of 50
    """
    if population_count <= 10:
        recommended = max(3, population_count // 3)
    elif population_count <= 30:
        recommended = max(3, population_count // 4)
    else:
        recommended = max(5, min(50, population_count // 5))
    
    return recommended

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in CSV or TSV format:',
                          st.session_state.textbox_content.strip(),
                          height=300,
                          key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    st.rerun()

# Replace the plot type radio to set Scatter Plot as default first option
plot_type = st.radio(
    "Plot Type:",
    ["Scatter Plot", "Tree Plot"],  # Scatter Plot first/default
    horizontal=True,
    help="Choose between t-SNE scatter plot or Ward hierarchical clustering tree"
)

# Handle plot type selection
if plot_type == "Tree Plot":
    # Add menu for standardization options
    standardize_option = st.selectbox(
        "Standardization:",
        [
            "None",
            "Standardize Rows",
            "Standardize Columns",
            "Standardize Both"
        ],
        help="Standardize data before clustering. Standardizing rows/columns subtracts mean and divides by std deviation."
    )
    plot_button = st.button("🌳 Plot Tree")  # Remove columns

# Simplify Scatter Plot options to just t-SNE with Plotly
elif plot_type == "Scatter Plot":
    # Get current population count to make a recommendation for perplexity
    population_count = 0
    if data_input:
        lines = [line for line in data_input.strip().split('\n') if line.strip()]
        population_count = len(lines)
    
    recommended_perplexity = calculate_recommended_perplexity(population_count)
    
    perplexity = st.number_input(
        f"Perplexity (recommended: {recommended_perplexity})",
        min_value=3, 
        max_value=100, 
        value=recommended_perplexity,
        help="Balance between local and global aspects. Lower values focus on local structure, higher values on global patterns."
    )
    
    plot_button = st.button("📈 Plot t-SNE")

# Simplify scatter plot code to just use t-SNE with Plotly
if plot_button and plot_type == "Scatter Plot":
    # Check for data first, before creating spinner
    if not data_input:
        st.warning(
            "Please add at least 3 populations before plotting.", icon="⚠️")
    else:
        with st.spinner("Creating t-SNE Plot..."):
            try:
                # Data preparation
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)

                # Check population count
                if not data.empty and len(populations) >= 3:
                    # Initialize t-SNE model with parameters
                    reduction_data = data.values
                    
                    # Use braycurtis as the fixed distance metric
                    distance_metric = "braycurtis"
                    
                    n_samples = len(populations)
                    # Adjust perplexity if sample count is small
                    perplexity_val = min(perplexity, n_samples - 1) if n_samples < perplexity + 1 else perplexity
                    
                    model = TSNE(
                        n_components=2,
                        perplexity=perplexity_val,
                        metric=distance_metric,  # Fixed to braycurtis
                        method='exact',  # For better accuracy with small datasets
                        learning_rate='auto'
                    )

                    # Perform dimensionality reduction
                    try:
                        result = model.fit_transform(reduction_data)
                    except Exception as e:
                        st.error(f"Error during t-SNE: {str(e)}")
                        st.info("Try a different perplexity value or check your data.")
                        st.stop()

                    # Create plot DataFrame
                    plot_df = pd.DataFrame(
                        data=result,
                        columns=[f'Dim1', f'Dim2']
                    )
                    plot_df['Populations'] = populations
                    
                    # Add t-SNE explanation
                    method_explanation = f"t-SNE visualizes population clusters using Bray-Curtis distance with perplexity {perplexity_val}."
                    if perplexity_val != perplexity:
                        method_explanation += f" (adjusted from {perplexity} based on population count)"
                    
                    st.caption(method_explanation)
                    
                    # Create scatter plot with Plotly
                    fig = px.scatter(
                        plot_df,
                        x='Dim1',
                        y='Dim2',
                        color='Populations',
                        title=f't-SNE Visualization (Bray-Curtis distance)',
                        text='Populations'
                    )
                    
                    # Customize plot
                    fig.update_traces(
                        textposition='top center',
                        hovertemplate='%{text}'
                    )
                    fig.update_layout(
                        legend_title_text='Populations',
                        xaxis_title="",
                        yaxis_title=""
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    
                else:
                    st.warning(
                        "Please add at least 3 populations before plotting.", icon="⚠️")

            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("Try adjusting the perplexity or check your data.")

# Update the tree plotting code - REMOVE STANDARDIZATION
if plot_button and plot_type == "Tree Plot":
    with st.spinner("Creating Tree..."):
        if data_input:
            try:
                # Remove leading/trailing whitespace and empty lines
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)
                
                # Check that there are more than 2 populations
                population_count = len(populations)
                if not data.empty and population_count > 2:
                    # Continue with tree plotting
                    labels = [i for i in populations]
                    height = max(20 * population_count, 500)
                    
                    # Convert data to float type to ensure numerical operations work
                    data_array = data.values.astype(float)
                    
                    # Standardization logic
                    standardization_caption = "Using raw data values (no standardization)"
                    if standardize_option == "Standardize Rows":
                        means = data_array.mean(axis=1, keepdims=True)
                        stds = data_array.std(axis=1, keepdims=True)
                        stds[stds == 0] = 1
                        data_array = (data_array - means) / stds
                        standardization_caption = "Standardized each row (population) to mean 0, std 1"
                    elif standardize_option == "Standardize Columns":
                        means = data_array.mean(axis=0, keepdims=True)
                        stds = data_array.std(axis=0, keepdims=True)
                        stds[stds == 0] = 1
                        data_array = (data_array - means) / stds
                        standardization_caption = "Standardized each column (feature) to mean 0, std 1"
                    elif standardize_option == "Standardize Both":
                        # Standardize columns first, then rows
                        means = data_array.mean(axis=0, keepdims=True)
                        stds = data_array.std(axis=0, keepdims=True)
                        stds[stds == 0] = 1
                        data_array = (data_array - means) / stds
                        means = data_array.mean(axis=1, keepdims=True)
                        stds = data_array.std(axis=1, keepdims=True)
                        stds[stds == 0] = 1
                        data_array = (data_array - means) / stds
                        standardization_caption = "Standardized columns, then rows (mean 0, std 1)"
                    
                    linkage_method = "ward"
                    distance_metric = "euclidean"
                    
                    distances = pdist(data_array, metric=distance_metric)
                    linkage_matrix = linkage(distances, method=linkage_method, metric=None)
                    
                    # Create dendrogram
                    fig = ff.create_dendrogram(
                        data_array,
                        orientation="right",
                        labels=labels,
                        distfun=lambda x: distances,
                        linkagefun=lambda x: linkage_matrix
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=height,
                        yaxis={'side': 'right'},
                        title="Hierarchical Clustering (Ward method)"
                    )
                    fig.update_yaxes(
                        automargin=True,
                        range=[0, len(populations)*10]
                    )
                    
                    # Add explanations
                    st.caption(standardization_caption)
                    st.caption("Ward clustering minimizes variance within groups, typically producing compact, balanced clusters.")
                    
                    # Display the dendrogram
                    st.plotly_chart(fig, theme=None, use_container_width=True, config={
                        'displayModeBar': True
                    })
                else:
                    st.warning(
                        "Please add at least 3 populations before plotting.", icon="⚠️")
            except Exception as e:
                st.error(f"Error creating tree plot: {str(e)}")
                st.info("Check your data.")