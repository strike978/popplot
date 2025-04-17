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

# Display the Textbox with the entire selected options
data_input = st.text_area('Enter data in CSV or TSV format:',
                          st.session_state.textbox_content.strip(),
                          height=300,
                          key='textbox_input')

# Check if the Textbox content has changed manually and clear session state if it has
if data_input != st.session_state.textbox_content.strip():
    st.session_state.textbox_content = data_input.strip()
    st.rerun()

# Replace the decomposition method selector with this new organization

# Update the plot type radio to set Scatter Plot as default first option
plot_type = st.radio(
    "Plot Type:",
    ["Scatter Plot", "Tree Plot"],  # Reordered to make Scatter Plot first/default
    horizontal=True,
    help="Choose between dimensionality reduction scatter plot or hierarchical clustering tree"
)

# Handle plot type selection
if plot_type == "Tree Plot":
    clustering_method = st.radio(
        "Tree Plot Method:",
        ["Ward", 
         "Complete (cityblock)",
         "Complete (braycurtis)"],  # Removed euclidean and minkowski
        horizontal=True,
        help="""
        Each method offers a different perspective on population relationships:
        
        Ward: Minimizes the variance of clusters (uses Euclidean distance).
        Complete (cityblock): Uses sum of absolute differences, less sensitive to outliers than Euclidean.
        Complete (braycurtis): Measures compositional dissimilarity, suitable for genetic abundance data.
        """
    )
    
    # Store selection in session state
    st.session_state.clustering_method = clustering_method
    plot_button = st.button("🌳 Plot")
# Update Scatter Plot options to make Isomap the default/first option
elif plot_type == "Scatter Plot":
    col1, col2 = st.columns([1, 1])  # Adjust columns since we're removing the distance metric option
    
    with col1:
        method = st.radio(
            "Scatter Plot Method:",
            ["Isomap", "t-SNE"],  # Reordered to make Isomap first/default
            horizontal=True,
            help="""
            Isomap: Preserves geodesic distances, good for visualizing population structure.
            t-SNE: Focuses on local clusters, can separate closely related populations.
            """
        )
    
    # Remove the distance metric selector since we're using braycurtis by default
    
    # Method-specific parameters
    if method == "t-SNE":
        perplexity = st.number_input(
            "Perplexity",
            min_value=3, 
            max_value=100, 
            value=30,
            help="Balance between local and global aspects. Lower values focus on local structure."
        )
    elif method == "Isomap":
        # Get current population count to make a recommendation
        population_count = 0
        if data_input:
            lines = [line for line in data_input.strip().split('\n') if line.strip()]
            population_count = len(lines)
        
        recommended_neighbors = calculate_recommended_neighbors(population_count)
        
        n_neighbors = st.number_input(
            f"Neighbors (recommended: {recommended_neighbors})",
            min_value=2, 
            max_value=min(100, max(20, population_count)),
            value=recommended_neighbors,
            help="Number of neighbors to consider for each point. Higher values capture global structure, lower values preserve local relationships."
        )
    
    st.session_state.decomposition_method = method
    plot_button = st.button("📈 Plot")

# Simplify scatter plot code by removing the identify clusters checkbox
if plot_button and plot_type == "Scatter Plot":
    # Check for data first, before creating spinner
    if not data_input:
        st.warning(
            "Please add at least 3 populations before plotting.", icon="⚠️")
    else:
        with st.spinner(f"Creating {method} Plot..."):
            try:
                # Data preparation
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)

                # Check population count
                if not data.empty and len(populations) >= 3:
                    # Initialize model based on selected method and parameters
                    reduction_data = data.values
                    
                    # Use braycurtis as the fixed distance metric for both methods
                    distance_metric = "braycurtis"
                        
                    if method == "t-SNE":
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
                    elif method == "Isomap":
                        # Recalculate recommended neighbors based on actual number of populations
                        actual_n_neighbors = min(n_neighbors, len(populations) - 1)
                        
                        # Create Isomap model with the neighbors parameter, but ensure it's valid
                        model = Isomap(
                            n_components=2,
                            n_neighbors=actual_n_neighbors,  # Use actual value, limited by population count
                            metric=distance_metric  # Fixed to braycurtis
                        )
                        
                        # Update explanation to reflect the actual value used
                        isomap_explanation = f"Isomap preserves geodesic distances using Bray-Curtis distance with {actual_n_neighbors} neighbors"
                        if actual_n_neighbors != n_neighbors:
                            isomap_explanation += f" (adjusted from {n_neighbors} based on population count)"

                    # Perform dimensionality reduction with appropriate data
                    try:
                        result = model.fit_transform(reduction_data)
                    except Exception as e:
                        st.error(f"Error during dimensionality reduction: {str(e)}")
                        st.info("Try a different method or metric.")
                        st.stop()

                    # Create plot DataFrame
                    plot_df = pd.DataFrame(
                        data=result,
                        columns=[f'Dim1', f'Dim2']
                    )
                    plot_df['Populations'] = populations
                    
                    # Create scatter plot
                    fig = px.scatter(
                        plot_df,
                        x='Dim1',
                        y='Dim2',
                        color='Populations',
                        title=f'{method} Visualization (Bray-Curtis distance)',  # Fixed to show braycurtis
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
                    
                    # Add method-specific explanations
                    method_explanation = ""
                    if method == "t-SNE":
                        method_explanation = f"t-SNE visualizes clusters by preserving local structure using Bray-Curtis distance with perplexity {perplexity}."
                    elif method == "Isomap":
                        method_explanation = isomap_explanation
                    
                    st.caption(method_explanation)
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                else:
                    st.warning(
                        "Please add at least 3 populations before plotting.", icon="⚠️")

            except Exception as e:
                st.error(f"Error creating plot: {str(e)}")
                st.info("Try a different method or check your data.")

# Update the tree plotting code to remove the even number requirement, keeping only the "more than 2" check
if plot_button and plot_type == "Tree Plot":
    with st.spinner("Creating Tree..."):
        if data_input:
            try:
                # Remove leading/trailing whitespace and empty lines
                cleaned_data_input = "\n".join(
                    line.strip() for line in data_input.splitlines() if line.strip())

                # Parse mixed format data
                populations, data = parse_input_data(cleaned_data_input)
                
                # Check only that there are more than 2 populations (removing the even number check)
                population_count = len(populations)
                if not data.empty and population_count > 2:
                    # Continue with tree plotting - no longer checking for even count
                    labels = [i for i in populations]
                    height = max(20 * population_count, 500)
                    
                    # Convert data to float type to ensure numerical operations work
                    data_array = data.values
                    
                    # Set the appropriate linkage method and distance metric based on selected method
                    selected_method = st.session_state.clustering_method
                    continue_standard_flow = True
                    
                    # Fix the Ward + Complete implementation to use Complete linkage with Ward-like characteristics
                    if selected_method == "Ward + Complete":
                        # For this hybrid approach:
                        # - Use Complete linkage (maximizes distances between clusters)
                        # - But with squared Euclidean distances (like Ward's method does)
                        # This combines Complete's ability to find distinct clusters with Ward's balanced approach
                        
                        # Calculate Euclidean distances (which Ward's method uses)
                        euclidean_distances = pdist(data_array, metric="euclidean")
                        
                        # Square the distances (this is what makes Ward's method minimize variance)
                        squared_distances = euclidean_distances**2
                        
                        # Use Complete linkage with these squared Euclidean distances
                        linkage_matrix = linkage(squared_distances, method="complete", metric=None)
                        
                        # Create the dendrogram using our hybrid approach
                        fig = ff.create_dendrogram(
                            data_array,
                            orientation="right",
                            labels=labels,
                            distfun=lambda x: squared_distances,
                            linkagefun=lambda x: linkage_matrix
                        )
                        
                        # Update the layout and add captions
                        fig.update_layout(
                            height=height,
                            yaxis={'side': 'right'}
                        )
                        fig.update_yaxes(
                            automargin=True,
                            range=[0, len(populations)*10]
                        )
                        
                        # Add method explanations for the hybrid approach
                        st.caption("Using Ward + Complete hybrid method for hierarchical clustering")
                        st.caption("This approach uses Complete linkage with squared Euclidean distances, combining Ward's variance minimization with Complete's distinct cluster separation.")
                        
                        # Display the dendrogram
                        st.plotly_chart(fig, theme=None, use_container_width=True, config={
                            'displayModeBar': True
                        })
                        
                        continue_standard_flow = False
                    
                    if continue_standard_flow:
                        # Further simplify methods to match the retained options
                        if selected_method == "Ward":
                            linkage_method = "ward"
                            distance_metric = "euclidean"
                        elif "cityblock" in selected_method:
                            linkage_method = "complete"
                            distance_metric = "cityblock"
                        elif "braycurtis" in selected_method:
                            linkage_method = "complete"
                            distance_metric = "braycurtis"
                        else:
                            # Default fallback
                            linkage_method = "ward"
                            distance_metric = "euclidean"
                        
                        # Compute distances if not already computed (only needed for standard metrics)
                        if 'distances' not in locals():
                            distances = pdist(data_array, metric=distance_metric)
                        
                        # Create linkage matrix
                        linkage_matrix = linkage(
                            distances,
                            method=linkage_method,
                            metric=None  # Already calculated distances
                        )
                        
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
                            yaxis={'side': 'right'}
                        )
                        fig.update_yaxes(
                            automargin=True,
                            range=[0, len(populations)*10]
                        )
                        
                        # Add explanations
                        st.caption(f"Using {selected_method} method for hierarchical clustering")
                        
                        if "cityblock" in selected_method:
                            st.caption("City-block distance sums the absolute differences across all genetic markers, less influenced by outliers.")
                        elif "braycurtis" in selected_method:
                            st.caption("Bray-Curtis distance measures compositional dissimilarity, identifying populations with similar genetic compositions.")
                        else:  # Ward
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
                st.info("Try a different method or check your data.")