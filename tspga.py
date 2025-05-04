import streamlit as st
import networkx as nx
import numpy as np
import random
import plotly.graph_objects as go

# ===================== Genetic Algorithm Functions ===================== #
def generate_initial_population(pop_size, n, start_node):
    """Generate random permutations starting with start_node"""
    return [[start_node] + random.sample([x for x in range(n) if x != start_node], n-1) 
            for _ in range(pop_size)]

def order_crossover(parent1, parent2):
    """Order Crossover (OX) for TSP"""
    n = len(parent1)
    start, end = sorted(random.sample(range(1, n-1), 2))
    offspring = parent1[start:end+1]
    
    current_pos = (end + 1) % n
    for gene in parent2:
        if gene not in offspring:
            if current_pos >= n:
                current_pos = 0
            offspring.insert(current_pos, gene)
            current_pos += 1
    return offspring[:n]

def mutate(path, mutation_rate):
    """Swap mutation"""
    path = path.copy()
    for i in range(1, len(path)-1):
        if random.random() < mutation_rate:
            j = random.randint(1, len(path)-2)
            path[i], path[j] = path[j], path[i]
    return path

def fitness(path, dist_matrix):
    """Calculate total path distance"""
    return sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1)) + dist_matrix[path[-1]][path[0]]

def tournament_selection(population, dist_matrix, tournament_size=3):
    """Tournament parent selection"""
    candidates = random.sample(population, tournament_size)
    return min(candidates, key=lambda x: fitness(x, dist_matrix))

def genetic_algorithm(dist_matrix, start_node, generations=500, pop_size=100, mutation_rate=0.1):
    """Optimized GA implementation"""
    n = len(dist_matrix)
    population = generate_initial_population(pop_size, n, start_node)
    best_path = min(population, key=lambda x: fitness(x, dist_matrix))
    best_fitness = fitness(best_path, dist_matrix)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for gen in range(generations):
        # Update progress
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {gen+1}/{generations} | Current Best: {best_fitness:.1f}")
        
        # Breed new population
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, dist_matrix)
            parent2 = tournament_selection(population, dist_matrix)
            child = mutate(order_crossover(parent1, parent2), mutation_rate)
            new_population.append(child)
        
        # Keep best solution
        population = sorted(population + new_population, key=lambda x: fitness(x, dist_matrix))[:pop_size]
        current_best = population[0]
        current_fitness = fitness(current_best, dist_matrix)
        
        if current_fitness < best_fitness:
            best_path = current_best
            best_fitness = current_fitness
    
    progress_bar.empty()
    status_text.empty()
    return best_path, best_fitness

# ===================== Plotly Visualization ===================== #
def plotly_tsp_graph(G, best_path=None):
    """Create interactive Plotly visualization for TSP"""
    pos = nx.spring_layout(G, seed=42)
    
    edge_traces = []
    node_x = []
    node_y = []
    node_text = []
    
    # Add all edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_traces.append(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add optimal path if available
    if best_path:
        path_edges = list(zip(best_path, best_path[1:] + [best_path[0]]))
        path_x = []
        path_y = []
        for edge in path_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
        
        edge_traces.append(go.Scatter(
            x=path_x, y=path_y,
            line=dict(width=2, color='red'),
            mode='lines',
            name='Optimal Path'
        ))
    
    # Add nodes
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line=dict(width=1, color='darkblue')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            uirevision='constant'
        )
    )
    
    # Add edge weight labels
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=(x0+x1)/2,
            y=(y0+y1)/2,
            text=str(edge[2]['weight']),
            showarrow=False,
            font=dict(size=8)
        )
    
    return fig

# ===================== Streamlit App ===================== #
st.set_page_config(page_title="TSP Solver", layout="wide")

# Initialize session state
if 'num_vertices' not in st.session_state:
    st.session_state.num_vertices = 7
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = np.array([
        [0,  2,  3,  4,  1,  37, 20],
        [2,  0,  2,  3,  7,  25, 37],
        [3,  3,  0,  3,  2,  23, 29],
        [3,  3,  2,  0,  2,  38, 22],
        [2,  2,  1,  2,  0,  15, 42],
        [28, 31, 13, 9, 22, 0,  42],
        [7, 37, 15, 33, 40, 34, 0]
    ], dtype=float)
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'best_path' not in st.session_state:
    st.session_state.best_path = None
if 'best_distance' not in st.session_state:
    st.session_state.best_distance = None

# Sidebar controls
st.sidebar.header("TSP Configuration")

# Vertex count selector
num_vertices = st.sidebar.number_input("Number of Vertices", min_value=2, value=7, step=1)
if num_vertices != st.session_state.num_vertices:
    st.session_state.num_vertices = num_vertices
    st.session_state.dist_matrix = np.zeros((num_vertices, num_vertices))
    st.session_state.graph = None
    st.session_state.best_path = None

# Matrix input
st.sidebar.subheader("Distance Matrix Input")
matrix_input = st.sidebar.text_area(
    f"Edit matrix ({num_vertices}x{num_vertices}):",
    value="[\n" + ",\n".join(["    " + str(row) for row in st.session_state.dist_matrix.tolist()]) + "\n]",
    height=300,
    key="matrix_editor"
)

# Matrix validation
try:
    new_matrix = eval(matrix_input)
    if isinstance(new_matrix, list) and len(new_matrix) == num_vertices:
        valid = True
        for row in new_matrix:
            if len(row) != num_vertices:
                valid = False
                break
        if valid:
            if all(new_matrix[i][i] == 0 for i in range(num_vertices)):
                if not np.array_equal(st.session_state.dist_matrix, new_matrix):
                    st.session_state.dist_matrix = np.array(new_matrix, dtype=float)
                    st.session_state.graph = None
                    st.session_state.best_path = None
            else:
                st.sidebar.error("Diagonal elements must be 0")
        else:
            st.sidebar.error(f"Matrix must be {num_vertices}x{num_vertices}")
    else:
        st.sidebar.error(f"Matrix must be {num_vertices}x{num_vertices}")
except Exception as e:
    st.sidebar.error(f"Invalid matrix format: {e}")

# Graph generation
if st.sidebar.button("Generate Graph"):
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            weight = st.session_state.dist_matrix[i][j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    st.session_state.graph = G
    st.session_state.best_path = None
    st.sidebar.success("Graph generated!")

# GA parameters
st.sidebar.subheader("Algorithm Parameters")
generations = st.sidebar.slider("Generations", 100, 2000, 500)
pop_size = st.sidebar.slider("Population Size", 50, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
start_node = st.sidebar.selectbox("Start Node", options=list(range(num_vertices)), format_func=lambda x: f"Node {x}")

# Main interface
st.header("Traveling Salesman Problem Solver - PFE Project")

# Visualization
if st.session_state.graph:
    fig = plotly_tsp_graph(
        st.session_state.graph,
        best_path=st.session_state.best_path
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Generate graph to see visualization")

# Controls and results
if st.button("Run Genetic Algorithm"):
    if st.session_state.graph is None:
        st.error("Please generate graph first!")
    else:
        with st.spinner("Optimizing TSP path..."):
            best_path, best_distance = genetic_algorithm(
                st.session_state.dist_matrix,
                start_node=start_node,
                generations=generations,
                pop_size=pop_size,
                mutation_rate=mutation_rate
            )
            
            st.session_state.best_path = best_path
            st.session_state.best_distance = best_distance
            st.rerun()

if st.session_state.best_path is not None:
    st.subheader("Results")
    st.write(f"**Best Path**: {' → '.join(map(str, st.session_state.best_path))} → {st.session_state.best_path[0]}")
    st.write(f"**Total Distance**: {st.session_state.best_distance:.2f}")

# Matrix validation info
st.sidebar.markdown("""
**Matrix Requirements**
- NxN numeric matrix (N = selected vertex count)
- Diagonal elements must be 0
- All other elements must be positive
""")

# Performance warning
if num_vertices > 15:
    st.sidebar.warning("Performance may degrade for graphs with more than 15 nodes")
